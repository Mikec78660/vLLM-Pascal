#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

// W4A16 decode GEMV for Pascal (sm_60/61, no FP16 tensor cores).
//   c[M,N] = a[M,K](fp16) @ dequant(b_packed[K,N/8])
// GPTQ sequential packing: int32 packs 8 cols, col j shift=(j%8)*4 (asymmetric).
// Each thread owns 8 consecutive cols over its split-K k-range, dequant ONCE,
// FMA into all M rows. split-K fills the P40s small SM count.
// SCALE_BF16 selects the scale element type (checkpoint scales are bf16).

template<int MAX_M, bool SCALE_BF16>
__global__ void w4a16_gemv_kernel(
    const __half* __restrict__ a,        // [M, K] fp16
    const int*    __restrict__ b_packed, // [K, N/8]
    const void*   __restrict__ scales,   // [K/G, N]  (fp16 or bf16)
    const int*    __restrict__ qzeros,   // [K/G, N/8]
    float*        __restrict__ partial,  // [S, MAX_M, N]
    int M, int N, int K, int G, int N8, int S)
{
  const int kchunk = K / S;
  const int k0 = blockIdx.y * kchunk;
  const int k1 = k0 + kchunk;
  const int n  = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (n >= N) return;
  const int n8 = n >> 3;

  float acc[MAX_M][8];
  #pragma unroll
  for (int m=0;m<MAX_M;m++)
    #pragma unroll
    for (int j=0;j<8;j++) acc[m][j]=0.f;

  float scale[8]; int zero[8];
  int gi = k0 / G;
  {
    #pragma unroll
    for (int j=0;j<8;j++){
      if (SCALE_BF16) scale[j] = __bfloat162float(((const __nv_bfloat16*)scales)[gi*N + n + j]);
      else            scale[j] = __half2float(((const __half*)scales)[gi*N + n + j]);
    }
    int z = qzeros[gi*N8 + n8];
    #pragma unroll
    for (int j=0;j<8;j++) zero[j] = (z >> (4*j)) & 0xF;
  }

  for (int k = k0; k < k1; k++) {
    if ((k & (G-1)) == 0 && k != k0) {
      gi = k / G;
      #pragma unroll
      for (int j=0;j<8;j++){
        if (SCALE_BF16) scale[j] = __bfloat162float(((const __nv_bfloat16*)scales)[gi*N + n + j]);
        else            scale[j] = __half2float(((const __half*)scales)[gi*N + n + j]);
      }
      int z = qzeros[gi*N8 + n8];
      #pragma unroll
      for (int j=0;j<8;j++) zero[j] = (z >> (4*j)) & 0xF;
    }
    float av[MAX_M];
    #pragma unroll
    for (int m=0;m<MAX_M;m++) av[m] = (m<M) ? __half2float(a[m*K + k]) : 0.f;
    int bp = b_packed[k*N8 + n8];
    #pragma unroll
    for (int j=0;j<8;j++) {
      float wf = ((float)((bp >> (4*j)) & 0xF) - (float)zero[j]) * scale[j];
      #pragma unroll
      for (int m=0;m<MAX_M;m++) acc[m][j] = fmaf(av[m], wf, acc[m][j]);
    }
  }
  #pragma unroll
  for (int m=0;m<MAX_M;m++)
    #pragma unroll
    for (int j=0;j<8;j++)
      partial[((blockIdx.y*MAX_M + m)*N) + n + j] = acc[m][j];
}

__global__ void reduce_partials_kernel(const float* __restrict__ partial,
                                       __half* __restrict__ c, int S, int M, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M*N) return;
  int m = idx / N, n = idx % N;
  float s = 0.f;
  for (int t=0;t<S;t++) s += partial[(t*M + m)*N + n];
  c[idx] = __float2half(s);
}

template<int MAX_M, bool SCALE_BF16>
void launch(const __half* a, const int* b, const void* sc, const int* z,
            float* partial, __half* c, int M, int N, int K, int G, int S,
            int block, cudaStream_t stream)
{
  int N8 = N/8;
  int threads_per_k = N/8;
  dim3 grid((threads_per_k + block - 1)/block, S);
  w4a16_gemv_kernel<MAX_M, SCALE_BF16><<<grid, block, 0, stream>>>(a,b,sc,z,partial,M,N,K,G,N8,S);
  dim3 rgrid((M*N + 255)/256);
  reduce_partials_kernel<<<rgrid, 256, 0, stream>>>(partial, c, S, M, N);
}

torch::Tensor w4a16_gemv(torch::Tensor a, torch::Tensor b_packed,
                         torch::Tensor scales, torch::Tensor qzeros,
                         int64_t group_size, int64_t splitk, torch::Tensor partial, int64_t block)
{
  TORCH_CHECK(a.is_cuda(), "cuda required");
  TORCH_CHECK(a.scalar_type()==at::kHalf, "activations must be fp16");
  TORCH_CHECK(a.is_contiguous() && b_packed.is_contiguous() && scales.is_contiguous() && partial.is_contiguous(), "contiguous");
  bool scale_bf16 = (scales.scalar_type()==at::kBFloat16);
  bool scale_fp16 = (scales.scalar_type()==at::kHalf);
  TORCH_CHECK(scale_bf16 || scale_fp16, "scales must be fp16 or bf16");
  int M = a.size(0), K = a.size(1), N8 = b_packed.size(1), N = N8*8;
  TORCH_CHECK(M <= 4, "decode GEMV supports M<=4");
  TORCH_CHECK(K % group_size == 0, "K % G != 0");
  int S = (int)splitk;
  TORCH_CHECK(K % S == 0 && (K/S) % group_size == 0, "bad splitk");
  TORCH_CHECK(partial.size(0)==S && partial.size(1)==4 && partial.size(2)==N, "partial shape mismatch");
  TORCH_CHECK(block==128 || block==256, "block must be 128 or 256");

  auto c = torch::empty({M, N}, a.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto ap=reinterpret_cast<const __half*>(a.data_ptr()); auto bp=reinterpret_cast<const int*>(b_packed.data_ptr());
  auto sp=scales.data_ptr(); auto zp=reinterpret_cast<const int*>(qzeros.data_ptr());
  auto pp=reinterpret_cast<float*>(partial.data_ptr()); auto cp=reinterpret_cast<__half*>(c.data_ptr());
  int Gi=(int)group_size;
  int Bl=(int)block;

  #define GO(mv, SB) launch<mv, SB>(ap,bp,sp,zp,pp,cp,M,N,K,Gi,S,Bl,stream)
  bool ok=false;
  if (M<=1) { if(scale_bf16){GO(1,true);ok=true;} else {GO(1,false);ok=true;} }
  else if (M<=2){ if(scale_bf16){GO(2,true);ok=true;} else {GO(2,false);ok=true;} }
  else if (M<=3){ if(scale_bf16){GO(3,true);ok=true;} else {GO(3,false);ok=true;} }
  else { if(scale_bf16){GO(4,true);ok=true;} else {GO(4,false);ok=true;} }
  TORCH_CHECK(ok,"unreachable");
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){ m.def("w4a16_gemv", &w4a16_gemv, "W4A16 decode GEMV"); }
