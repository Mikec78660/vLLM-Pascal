import os, time, torch, sys, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1"
sys.path.insert(0, "/tmp/w4a16gemv")

D = "/mnt/AI/1Cat-vLLM/venv/lib/python3.13/site-packages/vllm/model_executor/kernels/linear/mixed_precision"
CU = os.path.join(D, "w4a16_gemv.cu")

# --- Build via cpp_extension.load (reliable linking for the hand-built torch) ---
from torch.utils.cpp_extension import load
_m = load(
    name="w4a16_gemv",
    sources=[CU],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
)
w4a16_gemv = _m.w4a16_gemv
print("built + imported .so OK")
# place a copy next to the kernel for the server
import shutil
shutil.copy(_m.__file__, os.path.join(D, "w4a16_gemv.so"))
print("copied .so into venv:", os.path.join(D, "w4a16_gemv.so"))


def timeit(f, n=30):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


from safetensors.torch import load_file
ckpt = load_file("/AI/models/Qwen3.8-27B-AWQ-INT4/model-00003-of-00005.safetensors")
wq = ckpt["model.language_model.layers.22.mlp.gate_proj.weight_packed"].cuda().contiguous()
ws = ckpt["model.language_model.layers.22.mlp.gate_proj.weight_scale"].cuda().contiguous()
wz = ckpt["model.language_model.layers.22.mlp.gate_proj.weight_zero_point"].cuda().contiguous()
N, K8 = wq.shape
K = K8 * 8
N8 = N // 8
G = 32
Kg = K // G
print(f"gate_proj N={N} K={K} G={G} ckpt_scale_dtype={ws.dtype}")


def repack():
    sh = torch.arange(8, device=wq.device, dtype=torch.int32) * 4
    wu = ((wq.unsqueeze(-1) >> sh) & 0xF).reshape(N, K)
    wKN = wu.t().contiguous()
    qp = torch.sum((wKN.view(K, N8, 8) & 0xF) << sh, dim=2, dtype=torch.int32)
    return qp, ws.t().contiguous(), wz.t().contiguous()


qp, sc, zo = repack()
a = torch.randn(3, K, device="cuda", dtype=torch.float16)


def dequant_ref():
    sh = torch.arange(8, device=wq.device, dtype=torch.int32)*4
    # weights: wq [N,K/8], word k8 holds K-vals k8*8+j in nibble j
    w_un = ((wq.unsqueeze(-1) >> sh) & 0xF).reshape(N, K)          # [N,K] int32
    # zeros: wz [N/8,K/G], word n8 holds N-vals n8*8+j in nibble j
    z_un = ((wz.unsqueeze(-1) >> sh) & 0xF).permute(0,2,1).contiguous().reshape(N, K//G)       # [N,Kg]
    sc_full = ws.repeat_interleave(G, dim=1)                        # [N,K]
    z_full  = z_un.repeat_interleave(G, dim=1)                       # [N,K]
    return (w_un.float() - z_full.float()) * sc_full                 # [N,K] float

w_gt = dequant_ref()

for M in (1, 3):
    aM = a[:M].contiguous()
    for S in (1, 4, 8, 16, 32, 64):
        try:
            out = w4a16_gemv(aM, qp, sc, zo, G, S)
            ref = torch.mm(aM.float(), w_gt.T.float())
            err = (out.float() - ref).abs().max().item()
            ms = timeit(lambda: w4a16_gemv(aM, qp, sc, zo, G, S))
            gb = (qp.numel() * 4 + sc.numel() * 2 + zo.numel() * 4 + aM.numel() * 2 + out.numel() * 2) / 1e9
            st = "OK" if err < 0.05 else "FAIL"
            print(f"M={M} S={S:2d}  {ms:7.3f} ms  {gb/ms*1000:7.1f} GB/s  maxerr={err:.4f}  {st}")
        except Exception as e:
            print(f"M={M} S={S:2d}  ERROR {e}")
