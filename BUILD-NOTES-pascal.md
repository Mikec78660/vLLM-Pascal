# NOTE 2026-08-31: this file records the dev5-era upgrade (1.3.1.dev5+gf89434d9f).
# That wheel file and the /opt/1Cat-vLLM path no longer exist: the version is now the
# stable 1.3.1.dev0 (pascal-dev HEAD) and the runtime lives at /mnt/AI/1Cat-vLLM.
# The dev0 code is byte-identical to dev5 (git diff f89434d9f..978c95a4b -- vllm/ is empty),
# so every number in this file applies to the current runtime.

# vLLM-Pascal wheel rebuild log (2026-08-31, from clean clone)

Branch: `pascal-dev` on github.com/Mikec78660/vLLM-Pascal.
`pascal` is frozen at f9d2ec4ba. Every fix below is committed + pushed.

## Starting point

Full reset of llama-cpp-dev.lan (repo, mg-* merge trees, build venvs, /tmp
scraps) and re-clone of `pascal` from GitHub. Then: follow the README, fix
what breaks, commit each fix.

## Build environment (dev box)

- /opt/vllm-build-env (uv, py3.13.13, cmake 4.4.3, ninja, setuptools 77-81,
  setuptools-scm, setuptools-rust, wheel, jinja2, regex, protobuf, build)
- torch: /opt/torch-2.10.0-nccl.whl (source-built 2026-08-22, cu124,
  archs 6.0;6.1, NCCL in) staged at README path
  /opt/torch-src/pytorch/dist/torch-2.10.0-cp313-cp313-linux_x86_64.whl
- Flags: CUDA_HOME=/usr/local/cuda, TORCH_CUDA_ARCH_LIST="6.0;6.1",
  VLLM_SKIP_SM70_BUNDLES=1, VLLM_REQUIRE_RUST_FRONTEND=0,
  CMAKE_BUILD_TYPE=Release, MAX_JOBS=8, NVCC_THREADS=1

## What broke and why (evidence, not guesses)

### Failure 1 — GPTQ compile errors (32 in q_gemm.cu + qdq_*.cuh)

`__hfma2` undefined, `atomicAdd` ambiguous. WRONG first theory: "the GPTQ
source is CC7.0-only and needs patching". The source is fine — a plain
`nvcc -arch=sm_60` probe compiles all of it except `atomicAdd(half2*)`.

ROOT CAUSE: the f9d2ec4ba consolidation overwrote CMakeLists.txt + setup.py
with the UNPATCHED base. `CUDA_SUPPORTED_ARCHS` lost its `6.0;6.1` prefix,
so `CUDA_ARCHS = TORCH_CUDA_ARCH_LIST ∩ whitelist = EMPTY`, nvcc got NO
`-gencode`, and fell back to arch 52. Diagnosis signature (keep this):
CMakeCache `CMAKE_CUDA_ARCHITECTURES:STRING=52` + log line "pytorch is not
compatible with CMAKE_CUDA_ARCHITECTURES and will ignore" + FAILED nvcc
command with zero gencode flags.

### Failure 2 — CT W8A8-FP8 worker crash (boot of 9B-FP8-dynamic)

`RuntimeError: Expected lut_cuda.numel() == 256 && lut_cuda.is_cuda() ... got
false` in `mod.build_e4m3_lut(lut.cpu())`. "git == working prod code" had
copied the prod venv's `compressed_tensors_w8a8_fp8.py` verbatim — but that
file is the PARKED INT8-direct experiment: it set `_pascal_int8_direct=True`
by default (the GEMV path documented as corrupt-decode/blocked) AND passed a
CPU LUT to a binding that asserts CUDA.

META-LESSON: a byte-identical "working prod" venv file is NOT safe to commit.
Prod venvs accumulate parked experiments behind .bak files; the live default
may be the broken one.

## Commits (pascal-dev)

| commit | what |
|---|---|
| e75ce0a87 | P1 CMake arch whitelist +6.0;6.1 (both branches); P2 setup.py VLLM_SKIP_SM70_BUNDLES gate; P3 moe_wna16.cu half-atomicAdd shim (compat.cuh) |
| f89434d9f | W8A8-FP8 fallback: default = verified fp16-materialized path; INT8-direct GEMV opt-in behind VLLM_CT_W8A8_GEMV=1 with CUDA LUT |
| 6686687cd | README fixes (torchvision in Step 2, runnable smoke test, Step 0 torch-wheel reuse note) |

## Wheel

`1cat_vllm-1.3.1.dev5+gf89434d9f.cu124-cp313-cp313-linux_x86_64.whl`
(45 MB, /opt/1Cat-vLLM/dist/, relayed as llama.lan:/tmp/vllm-dev5.whl)

Verified inside the wheel:
- 5 GEMV .so (w4a16, w8a16, w8a16_v8, w8a16_v11, w16a16) + 41 triton_kernels
- SASS sm_60/sm_61 only (cuobjdump: _C_stable 26+26, _C 7+7; no sm_7*)
- 1334 gptq symbols in _C_stable (matches prod)
- fp8->int8 KV alias in cache.py
- WNA16 capability gates (70/75->60), W8A8 dequant fallback, eager backend
- No SM70 bundles (flash_attn_v100.py stub + flash_qla python remain by design;
  no SM70 .so/.cu in the wheel)

## Deployment (llama.lan)

- Fresh venv /tmp/vllm-test/venv (uv, py3.13.5): wheel --no-deps + torch
  --no-deps + nvidia-* pins from torch METADATA + triton==3.6.0 + 68-line
  curated runtime reqs (/root/v100build/vllm-runtime-requirements.txt) +
  torchvision==0.25.0 --no-deps + ld.so.conf.d/vllm-test.conf.
- Gotcha hit: uv rejects wheels without a valid filename (needs python/ABI
  tags) — copy to a proper name before `uv pip install`.
- Gotcha hit: model registry needs torchvision for Qwen3.5 VL archs.
- Smoke: import ok, cc 6.1, attention backend resolves TRITON_ATTN.

## Benchmark result (9B-FP8-dynamic, P40 idx 2,3, TP=2, ctx 32768, KV fp16,
graphs FULL, warm triton cache)

| conc | dev5 wheel | README baseline |
|---|---|---|
| x1  | 23.52 | 23.35 |
| x2  | 17.21 | 17.65 |
| x4  | 32.76 | 32.38 |
| x8  | 51.02 | 49.14 |
| x16 | 65.64 | 63.34 |
| prefill 2439 tok | 233 t/s | 213.7 t/s |

KV pool 690,113 tokens (20.4x @ 32k). Coherence: Asimov's three laws exact.
Dequant fallback log: "CT W8A8-FP8 Pascal dequant fallback active
(capability 61)". Result JSON: llama.lan:/root/vllm-pascal-results/.
NOTE: the first quick client (bench9b.py) faked x2..x16 aggregates (1
request x N); bench2.py (canonical parallel client) numbers above are the
real ones.

## Open items

- 27B-INT8 TP=4 re-verification needs GPUs 0-3 (currently in use by the user);
  9B-TP2 validation exercises build + deploy + CT-W8A8 + TRITON_ATTN + NCCL
  but not the Exllama WNA16 int8 path.
- Integrate pascal-dev -> pascal when the user approves (pascal is frozen).
- GPTQ checkpoints (quant_method=gptq) are now compilable but untested at
  runtime on this rig (no GPTQ checkpoint in use).
