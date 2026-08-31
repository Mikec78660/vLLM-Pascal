# vLLM-Pascal

> **Upstream:** [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM.git) — a vLLM
> engineering fork built for **Tesla V100 / SM70** (TurboMind-derived SM70 kernels,
> V100 FlashAttention path, long-context runtime defaults, AWQ on Volta).
>
> **This fork** — vLLM-Pascal, a sibling of that project — makes the same class of
> models run on **Pascal GPUs (CC 6.x): Tesla P40 (6.1) and P100 (6.0)**. Pascal has
> no tensor cores for FP16/BF16, no FP8, no native INT8 tensor ops, and its Triton
> codegen path rejects `fp8e4nv` outright. Every kernel in this fork's decode path
> was measured on real P40/P100 hardware, and the numbers below are what the
> hardware actually does.

vLLM-Pascal is a branch of the 1Cat-vLLM repo. It adds CC 6.0/6.1 to the build
(`TORCH_CUDA_ARCH_LIST="6.0;6.1"`), skips the SM70-only kernel bundles, and patches
the kernel/config gaps that otherwise make vLLM unusable or ~5× slow on Pascal.
All Qwen3.5/3.8 checkpoints tested here are hybrid GDN linear-attention models
(48 of 64 layers GDN, 16 full attention) — the hardest case for a kernel stack.

## What the fork changes (Pascal-relevant patches, newest first)

| Patch | What it does | Why |
|---|---|---|
| `kv_cache_dtype fp8 → int8_per_token_head` alias  | On CC < 70, `--kv-cache-dtype fp8` is remapped to `int8_per_token_head` | Triton on SM6.x cannot compile `fp8e4nv`; the int8 per-token-head path uses the same 1 byte/token, has better precision (7-bit mantissa), and is fully supported by the TRITON_ATTN backend. Your harness keeps passing `fp8`; the fork quietly does the right thing. |
| CT W8A8-FP8 load-time dequant fallback  | `compressed-tensors` per-channel FP8 (e.g. RedHatAI `*-FP8-dynamic`) dequants to FP16 at load on CC < 70; decode runs cuBLAS FP16 | Upstream scheme hard-gates `min_capability 89` (cutlass scaled_mm, SM89 kernels). Pascal: materialize weights once, never touch FP8 at runtime. |
| Triton W4A16 prefill block-cliff fix  | M>64 prefill now uses (32,32,32) blocks on SM6.x | Upstream (128,128,32) picks spill registers on Pascal → ~37 GF/s. A 151-token prompt went from **734 ms → 13 ms** (55×). |
| W4A16 split-K CUDA GEMV (`w4a16_gemv.cu`) | Custom decode GEMV: each thread owns 8 output columns, dequants each packed int32 nibble **once**, FMAs into all M≤4 accumulator rows; int4 weights stay resident | Pascal has no tensor cores: the stock Triton path re-dequantizes weight tiles per activation row (~10× wasted ALU). 1.69 → 0.24 ms per GEMM at M=1 (213 GB/s, near P40 GDDR5 roofline). |
| NCCL all_gather via pynccl | Routes TP `all_gather` through `ncclAllGather` (SHM transport on this box) instead of the gloo CPU fallback | The source-built torch has `USE_NCCL=0`; without this, every TP step round-trips GPU→host→gloo→host→GPU. |
| Qwen3.5 MTP support (`qwen3_5_mtp.py` + spec config wiring) | Native MTP head as speculative draft model; `--speculative-config '{"method":"mtp",...}'` | Upstream rewires `qwen3_5` to `qwen3_5_mtp` only in specific paths; this fork wires the named `mtp` method for Pascal. |
| DFlash2-as-V1 bridge  | Checkpoint-format bridge so parallel-draft checkpoints load | The *decoding* (non-causal attention) is hard-blocked on Pascal — FlexAttention path dies on hybrid GDN block sizes. Format-only support. |
| CMake: `CUDA_SUPPORTED_ARCHS "6.0;6.1;..."` (two branches) + `VLLM_SKIP_SM70_BUNDLES=1` | Build for Pascal, don't build SM70-only bundles (FA-V100, flash_qla, tcmalloc) | The upstream V100 branch's bundles are SM70-only and waste the build. |

Also in this fork: FP8 blockwise ([128,128]) dequant fallback, Exllama WNA16
routing for int8 checkpoints, CUDA-graph-safe partial-buffer caching, tool-call
parser `qwen3_xml` for Qwen3.x instruct models.

Provenance note: the same patches also exist on the preserved experimental
v1.3.0-merged line (tag `legacy/a561ea8a1-preconsolidate`); this branch
carries them as hand-tuned files, not that commit history.

## Test system

All numbers below come from one physical machine (`llama.lan`), a Debian 13 (trixie)
LXC container on a Proxmox host:

| | |
|---|---|
| CPU | AMD EPYC 3151 — 4 cores / 8 threads, 1.2–2.7 GHz, 16 MiB L3 |
| Disk | **28 GB root FS** (too small for the 8 GB venv) + **1 TB data disk** at `/mnt/AI` → the runtime lives at **`/mnt/AI/1Cat-vLLM`** (one path, no aliases, no second copies) |
| RAM | 24 GiB (LXC cgroup), + 8 GiB swap; 16 GiB tmpfs `/tmp` for venv/caches |
| GPUs | **5× Tesla P40** (24 GB GDDR5, ~196 GB/s, CC 6.1, idx 0–4) + **2× Tesla P100-PCIE** (16 GB HBM2, ~732 GB/s, CC 6.0, idx 5,6) |
| PCIe | **Gen1 (2.5 GT/s)**. Slots are x16 but negotiate **x1** in this chassis → ~250 MB/s per GPU. Model loads are PCIe-bound (a 13 GB dequantized 9B takes ~52 s); TP collectives run over shared memory (NCCL SHM transport, P2P enabled between the P40s), and decode is GPU-bandwidth-bound, so the x1 link does **not** limit tok/s — only load time. |
| GPU topology | All 7 cards share one PCIe switch (PIX), single NUMA node |
| Driver / runtime | NVIDIA driver 550.x (host passthrough), Python 3.13, uv venv, **Debian `nvidia-cuda-toolkit` 12.4** for the cu12 runtime libs, + **one** `ld.so.conf.d` entry exposing the venv's pip `nvidia_cufile` (torch NEEDs `libcufile.so.0`; the toolkit doesn't ship it). No `/usr/local/cuda`, no `nvcc` — Triton 3.6.0 ships its own `ptxas`; the source-built torch cu124 wheel resolves the **system** cu12 libs via its RUNPATH. |

## Models tested

| Model (checkpoint) | Quant | Size | Notes |
|---|---|---|---|
| `Qwen3.5-9B-FP8-dynamic` | compressed-tensors, per-channel FP8 (e4m3) | 14 GB | RedHatAI. On Pascal → FP16-materialized fallback (26 GB resident). No MTP head in this checkpoint. |
| `Qwen3.5-9B-w4a16` | compressed-tensors WNA16 int4, **symmetric** | 11 GB | RedHatAI. 232 modules stay BF16 (GDN/vision/embeddings); symmetric ⇒ the asymmetric GEMV gate doesn't fire, Exllama kernel serves it. |
| `Qwen3.8-27B-AWQ-INT4` | AWQ int4, asymmetric | 20 GB | The GEMV showcase: near-total quant coverage, int4-resident weights. |
| `Qwen3.8-27B-FP8` | blockwise [128,128] FP8 | 29 GB | On Pascal → FP16-materialized. Has native MTP head. |
| `Qwen3.8-27B-INT8-W8A16-MTP` | compressed-tensors group-128 symmetric int8 | 30 GB | True weight-only W8A16; BF16 kept for vision/lm_head/MTP head. Served by the Exllama WNA16 kernel (no custom kernel needed). |

All launches: `--dtype float16`, CUDA graphs (`--compilation-config
'{"mode":0,"cudagraph_mode":"FULL"}'` — the GDN backend downgrades to
FULL_DECODE_ONLY, i.e. decode is graphed, prefill eager), `--enable-auto-tool-choice
--tool-call-parser qwen3_xml`, `NCCL_P2P_DISABLE=0`. KV cache dtype `fp16` unless
stated. **Never pass `--enforce-eager`** — eager re-dispatches ~128 kernels per
decode token through Python and burns 4 CPU cores; with graphs worker CPU is ~0%.

## Benchmark method

- Canonical client: 250-token generation, temp 0, `ignore_eos`, **1 warmup request
  first** (absorbs first-call Triton JIT + DVFS ramp — P40s idle at P8 state and
  ramp ~1 s after load; un-warmed numbers can read 30% low), token counts from
  `usage.completion_tokens` (MTP streams multiple tokens per SSE chunk — chunk
  counts are wrong).
- Decode t/s per stream = tokens/wall; aggregate = sum over N parallel streams.
- Prefill = time-to-first-token over a 2048-token prompt, t/s = prompt_tokens/TTFT.
- KV pool = the engine's own `GPU KV cache size` log line (trust it; the graph
  memory-profiling accounting shifts effective budget between boots).
- **First pass on a fresh Triton cache is a warm-up, not a measurement.** With a
  cold cache the first requests trigger in-situ kernel JIT (visible as `Triton
  kernel JIT compilation during inference` warnings inside the bench window) and
  the numbers come out 1.5–3× low. All measured numbers below are from warm-cache
  runs; contaminated first-pass values are kept only as a warning, not data.
- **One server at a time.** Two vLLM servers benchmarking concurrently on this box
  interfere through the shared CPU (8 threads) and the shared PCIe/SHM path; all
  tables below are sequential.

## Results

### Qwen3.5-9B-FP8-dynamic (→ FP16 materialized) — the P40 x2 / x4 matrix

P40s 0,1 (TP=2), ctx 32768, KV fp16, graphs — full sweep, warm cache:

| conc | decode t/s (agg) | per stream |
|---|---|---|
| x1 | 23.35 | 23.35 |
| x2 | 17.65 | 8.8 |
| x4 | 32.38 | 8.1 |
| x8 | 49.14 | 6.1 |
| x16 | **63.34** | 4.0 |

KV pool 651,884 tokens (19.9× @ 32k); prefill 213.7 t/s (2.4k prompt); weights
8.91 GiB/card. Aggregate climbs past x8: at TP=2 per-card weight bytes are half
the single-card case and GDDR5 streaming stays the binding limit until ~x16.
TP=1 (single P40, GPU 4): x1 15.24 / x4 22.98 / x8 43.05, KV pool only 59,578
(1.82× @ 32k) — the dequantized 17.7 GiB leaves little VRAM; fine for a few
sessions, not a production shape. TP=4: x1 28.78 / x4 41.44 / x16 61.9, pool
1,860,825 (56.8× @ 32k) — same ceiling as TP=2 (weight streaming), just more KV.

Context-size effect, P40 pair (TP=2): **decode speed does not depend on ctx size**
(the hybrid GDN state is fixed per sequence; only 16/64 layers scale with tokens).
What ctx size changes is **capacity**: at 262k ctx the fp16-KV pool is
730,333 tokens (2.79× @ 262k) while decode stays 23.37 t/s (x1) / 32.07 (x4) —
identical to the 32k numbers above. ctx costs nothing at decode, everything at
prefill and capacity.

### Qwen3.5-9B on P100s (idx 5,6, TP=2) — P40 vs P100

Same checkpoint, same flags, ctx 32768, KV fp16, graphs, warm cache:

| conc | P40 x2 (0,1) | P100 x2 (5,6) | delta |
|---|---|---|---|
| x1 | 23.35 t/s | 27.88 t/s | +19% |
| x2 | 17.65 t/s | 20.23 t/s | +14% |
| x4 | 32.38 t/s | 36.91 t/s | +14% |

KV pool on the 16 GB P100 pair @ 32k: 141,001 tokens (4.3×) vs 651,884 on the
24 GB P40 pair.

Why the P100's ~3.7× bandwidth advantage only buys 15–22%: at TP=2 every decode
step pays an NCCL allreduce; on the P40s weight streaming dominates so the
collective is a small fraction of step time, on the P100s weights stream so fast
that the fixed per-step sync cost becomes the binding limit (Amdahl on the
collective). KV pool on the 16 GB P100 pair @ 32k: 226,397 tokens (6.9×); @ 8k:
122–197k depending on boot.

### Qwen3.5-9B-w4a16 (TP=2 P40s)

This fork (prod wheel):

| conc | x1 | x2 | x4 | x8 |
|---|---|---|---|---|
| decode t/s (agg) | 11.2 | 13.9 | 18.1 | 19.5 |

Prefill 27.7 t/s (2.4k prompt). KV pool 909,560 (27.8× @ 32k), weights
5.25 GiB/card.

The symmetric checkpoint is **slower than its own FP8→FP16 sibling** (11.2 vs
23.4 at x1): no qzeros ⇒ the custom GEMV's asymmetric gate fails, and 232
BF16-retained modules mean int4 saves less bandwidth than expected while dequant
ALU stays on the batch-1 critical path. INT4 only wins with near-total quant
coverage (see 27B AWQ below).

### Qwen3.8-27B-AWQ-INT4 (TP=2 P40s, +MTP k=2)

| | before GEMV fix | with W4A16 GEMV (this fork) | llama.cpp Q4_0+MTP (same 2 cards) |
|---|---|---|---|
| x1 | 4.08 t/s | **13.2 t/s** | 13.6–13.7 t/s |

The 4.08 was the engine, not the cards: eager dispatch + the stock Triton
dequant-ALU GEMM. With the split-K CUDA GEMV (7× per-GEMM at M=1) and graphs,
vLLM meets llama.cpp on the same hardware. The prefill block-cliff fix
(48322deed) is what makes AWQ *prefill* usable at all (151-tok prompt:
734 ms → 13 ms).

(The experimental v1.3.0-merged wheel routed this checkpoint to
`TritonW4A16LinearKernel` at 11.23 x1; that wheel is not in this branch —
see the dead-ends note below for what that gap actually was.)

### Qwen3.8-27B-FP8 (TP=4 P40s, +MTP k=2, graphs)

| conc | decode t/s (agg) | per stream |
|---|---|---|
| x1 | 8.70 | 8.70 |
| x2 | 12.51 | 6.3 |
| x4 | 19.04 | 4.8 |

(warm cache; earlier same-flags runs: x1 7.41–7.57, x4 ~16.4–17.6 — the spread
is the MTP acceptance variance between boots, not a config difference)

Aggregate saturates ~16–19 t/s; the 3rd session is near-free, the 4th adds
little; more sessions queue. FP8 on Pascal buys **footprint, not speed**: the
dequant fallback leaves FP16 weights (2 bytes/weight) so the decode ceiling is the
weight-streaming floor (~12 t/s best-case single). KV pool: 336,855 tokens @
util 0.92 (10.3× @ 32k; earlier boots: 290k @ 0.85, 422k @ 0.92). MTP k=2
acceptance ~80% (position 1 ~0.82) — a real but modest win on this checkpoint;
`max_num_scheduled_tokens` auto-sets to 2048.

### Qwen3.8-27B-INT8-W8A16 (TP=4 P40s, ctx 32768)

| config | x1 | x2 | x4 | x8 |
|---|---|---|---|---|
| eager (no MTP) | 15.28 | 22.95 | 29.21 | 29.27 |
| **graphs (no MTP) — production (this wheel)** | **16.67** | — | **30.94** | 27.51 |
| graphs + MTP k=2 | 7.92 | 9.88 | 9.21 | — |

KV pool 769,683 tokens (23.5× @ 32k), weights 7.49 GiB/card, prefill 11.2 t/s
(2.4k prompt).

- int8 weights = half the bytes of the FP16-materialized FP8 build ⇒ **+24% x1 /
  +11% x8** — exactly the halved-bandwidth prediction. Served by the stock Exllama
  WNA16 kernel (no custom kernel needed).
- MTP is a **loss** on this checkpoint even with graphs: acceptance 63–65% (K=2),
  and the GDN draft loop can't be graphed so decode reverts to eager. K=1 gives
  no meaningful single-stream gain. **Run it with no MTP.**
- KV pool **815,559 tokens (24.9× @ 32k)** — int8 weights leave enormous VRAM for
  KV. x32 aggregate ≈ 29.3 (identical to x8: the server saturates past 4 streams).
- Coherence verified: Asimov's three laws of robotics recited exactly.

### FP8 KV alias on Pascal (this fork's signature feature)

`--kv-cache-dtype fp8` on CC 6.x would normally die in the Triton compiler
(`type fp8e4nv not supported in this architecture`). The fork remaps it to
`int8_per_token_head` at config validation (one log line, no harness changes):

| 9B FP8-dynamic TP=2 P40s, ctx 262144, util 0.95 | KV pool | @ 262k |
|---|---|---|
| `fp16` KV | 730,333 tokens | 2.79× |
| `fp8` (aliased → int8 per-token-head) | **1,425,408 tokens** | **5.44×** |

Decode is identical across both (x1 23.37 vs 23.49, x4 32.07 vs 30.26) — the
alias costs nothing at decode and roughly doubles 262k-session capacity.
Verified identical pool between `--kv-cache-dtype fp8` and
`--kv-cache-dtype int8_per_token_head` (byte-identical config path), and a
41k-token coherence test passed on the aliased server.

## Dead ends (do not retry; full write-ups in the skill)

- `--dtype float32`: OOM at profile on P100s (144 MiB alloc on 15.9 GiB); on P40s
  it boots then dies on first request — `ChunkGatedDeltaRuleFunction does not
  support float32`. **fp16 is the only working dtype on CC 6.x.**
- **The "2.7× Exllama-v2 regression" was a benchmark-harness artifact, not a
  kernel regression.** The experimental v1.3.0-merged wheel (dev388) showed
  27B-INT8 at 6.18 vs 16.67 and 9B-W4A16 at 7.95 vs 11.2 x1. A controlled 2×2
  A/B matrix (prod vs test venv × `language_model_only` on/off, identical
  flags, same GPU pair, warm cache) showed the gap came from the comparison
  runs using different serving configs (`language_model_only=True` +
  `max_num_seqs=4` in prod vs none + `max_num_seqs=8` in the test harness),
  not from kernels — `exllama.py` is **byte-identical** (same sha256) across
  the pre-merge, merged, and prod trees. The merged wheel was therefore not
  kept; this branch *is* the working prod tree. The merged line is preserved
  as tag `legacy/a561ea8a1-preconsolidate` for reference.
- FP8-dynamic and plain-fp16
  checkpoints are unaffected (they don't touch the WNA16 path).
- INT8-direct byte-resident GEMV (`w8a16_gemv_v8`): kernel proven bit-exact
  offline and per-call, but server integration degenerates decode deterministically
  at the first ~2 linears of each step (a second consumer reads the weight bytes).
  Blocked, fully characterized.
- DFlash2 parallel drafting: hard-blocked (non-causal attention only exists in
  FlexAttention, which dies on hybrid GDN block sizes).
- W16A16 split-K GEMV for the FP8 dequant path: unvalidated; FP8 TP=4 stays at the
  cuBLAS-FP16 floor (~12 t/s ceiling single).
- P100 fp8e4nv KV: same Triton rejection as P40 — the alias is the answer for both.

## Clone & build

### Prerequisites (build machine — any 28-core/27 GB x86 box, no GPU needed)

- Ubuntu 24.04 / Debian 13, gcc 12.2+, `apt install cuda-toolkit-12.4` (compiler
  only; `CUDA_HOME=/usr/local/cuda`), uv, git.
- Python 3.13.

### Step 0 — torch from source (one-time; ~1 h on 28 cores)

**If you already have the wheel, skip this step.** The source-built wheel
`torch-2.10.0-cp313-cp313-linux_x86_64.whl` (cu124, archs 6.0;6.1, NCCL
included, built 2026-08-22) lives at
`llama-cpp-dev.lan:/opt/torch-2.10.0-nccl.whl` — just copy it to
`/opt/torch-src/pytorch/dist/` to match the Step 1 layout. Rebuild from
source only if you need a different arch list or NCCL config.

vLLM pins `torch==2.10.0`; the PyPI wheel is cu128 and won't run on our
550-series driver, so we build torch once with cu124 + Pascal archs:

```bash
git clone -b v2.10.0 --depth 1 https://github.com/pytorch/pytorch /opt/torch-src/pytorch
uv venv --python 3.13 /opt/torch-build-env
# the FULL pyproject [build-system] requires list — missing `packaging` is a
# SILENT total killer (gen_torch_version CMake step dies every attempt):
uv pip install --python /opt/torch-build-env/bin/python \
    cmake ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" \
    "setuptools-scm>=8.0" wheel jinja2 requests "typing-extensions>=4.10.0" six
cd /opt/torch-src/pytorch
# build WITH NCCL (the old recipe used USE_NCCL=0 — that is the root of the
# vLLM gloo-fallback CPU burn):
CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=1 \
TORCH_CUDA_ARCH_LIST="6.0;6.1" \
PYTORCH_BUILD_VERSION=2.10.0 \
python setup.py bdist_wheel
# → dist/torch-2.10.0-cp313-*.whl (355 MB). OOM-aware retry: halve
# CMAKE_BUILD_PARALLEL_LEVEL on code=137; the build tree is resumable.
```

### Step 1 — build the vLLM-Pascal wheel

```bash
# ONE canonical runtime path. On llama.lan it MUST be /mnt/AI/1Cat-vLLM:
# the root FS is 28 GB (venv is 8 GB) and /mnt/AI is the 1 TB data disk.
# A fresh LXC with a normal root disk may use /opt/1Cat-vLLM instead.
# Pick ONE and never create a second copy or alias of it.
export RT=/mnt/AI/1Cat-vLLM
git clone -b pascal-dev https://github.com/Mikec78660/vLLM-Pascal $RT
cd $RT   # branch pascal-dev: the working, consolidated code
uv venv --python 3.13 /opt/vllm-build-env
uv pip install --python /opt/vllm-build-env/bin/python \
    cmake ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" \
    "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2 regex protobuf build
# the custom torch wheel (satisfies the fork's torch==2.10.0 pin) + build-time deps:
uv pip install --python /opt/vllm-build-env/bin/python --no-deps \
    /opt/torch-src/pytorch/dist/torch-2.10.0-cp313-*.whl
uv pip install --python /opt/vllm-build-env/bin/python \
    numpy sympy networkx filelock typing-extensions fsspec

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="6.0;6.1"      # P100 + P40 — the whole point
export VLLM_SKIP_SM70_BUNDLES=1            # don't build SM70-only bundles
export VLLM_REQUIRE_RUST_FRONTEND=0
export CMAKE_BUILD_TYPE=Release
MAX_JOBS=8 NVCC_THREADS=1 \
/opt/vllm-build-env/bin/python setup.py bdist_wheel
# OOM (27 GB box): on code=137 halve MAX_JOBS (floor 2), NVCC_THREADS=1.
# → dist/1cat_vllm-1.3.1.dev0+cu124-cp313-cp313-linux_x86_64.whl
# (the version is pinned to a stable `1.3.1.dev0` in setup.py — every rebuild of
#  this branch yields the SAME wheel name; provenance is the git commit, not
#  the version. Verify it is Pascal-only: cuobjdump the .so, expect sm_60/sm_61
#  only, zero sm_70/sm_52.)
```

Key compile facts: `setup.py` imports torch at module top ⇒ `--no-build-isolation`
(or the venv approach above) is mandatory; `VLLM_SKIP_SM70_BUNDLES=1` wraps the
unconditional SM70 bundle steps; CMake's `CUDA_SUPPORTED_ARCHS` is patched in
**two** if-branches (CUDA-12 path + fallback); the wheel embeds only sm_60/61
(verify: `python -c "import vllm._C"` + `strings vllm/_C.abi3.so | grep -c sm_7` → 0).

### Step 2 — deploy on the GPU host

A fresh LXC needs **two things it won't have by default**: the NVIDIA
**driver** (550.x — free with the GPU passthrough) and the **CUDA 12.4
runtime libs**. The runtime libs are the catch. The source-built torch wheel
bundles **no** nvidia `.so` and its METADATA declares **no** nvidia deps —
`libtorch_cuda.so` has `RUNPATH $ORIGIN:/usr/local/cuda/lib64` plus
`NEEDED libcublas.so.12` / `libcufft.so.11` / `libcurand.so.10` /
`libcusparse.so.12` / `libnvJitLink.so.12`, so it resolves those from the
**system** ld path. On llama.lan that is the Debian `nvidia-cuda-toolkit
12.4` package. Install the matching runtime libs on any fresh LXC (driver
550.x caps at CUDA 12.4, so use the 12.4 series):

    apt-get install -y nvidia-cuda-toolkit   # 12.4: libcublas12 libcublaslt12 \
                                             # libcufft11 libcurand10 \
                                             # libcusparse12 libcusolver11 \
                                             # libnvjitlink12
    # verify torch can resolve them — every line must resolve, none "not found":
    ldd $RT/venv/lib/python3.13/site-packages/torch/lib/libtorch_cuda.so \
      | grep -E 'libcublas.so.12|libcufft.so.11|libcurand.so.10|libcusparse.so.12|libnvJitLink.so.12'

No `nvcc` / `/usr/local/cuda` is needed at runtime: Triton 3.6.0 ships its
own `ptxas` and the wheel is prebuilt.

Then build the venv. **Everything is `--no-deps`** — a bare install lets the
resolver pull the PyPI cu128 torch build and clobbers the hand-built Pascal
wheel:

    uv venv --python 3.13 $RT/venv
    V=$RT/venv/bin/python
    # 1. the vLLM-Pascal wheel you built in Step 1:
    uv pip install --python $V --no-deps \
        /path/to/1cat_vllm-1.3.1.dev0+cu124-cp313-cp313-linux_x86_64.whl
    # 2. the source-built torch wheel from Step 0:
    uv pip install --python $V --no-deps \
        /path/to/torch-2.10.0-cp313-cp313-linux_x86_64.whl
    # 3. the curated runtime deps — committed in this repo (the exact set the
    #    known-good venv was built from; all py3-none/cp313 wheels, no compile):
    uv pip install --python $V --no-deps -r requirements/runtime_pascal.txt
    # (for a BYTE-IDENTICAL venv: use the exact-version freeze instead —
    #  uv pip install --python $V --no-deps -r requirements/runtime_pascal_freeze.txt
    #  torch + the vLLM wheel come from the local wheels in Step 0/1, not PyPI)
    # 4. torchvision is REQUIRED for the Qwen VL model files: the registry
    #    inspects the arch in a subprocess that imports
    #    image_processing_qwen2_vl -> torchvision.transforms at module import.
    #    Missing it dies at boot ("Error in inspecting model architecture"):
    uv pip install --python $V --no-deps torchvision==0.25.0
    # 5. (optional) NIXL KV-cache-transfer connector — needed only if you run
    #    disaggregated prefill/decode with `--kv-transfer-config` NixlConnector.
    #    Install `nixl` (the connector backend) + the cu12 backend. Do NOT use
    #    the bare `nixl` meta alone: it hard-depends on BOTH nixl-cu12 and
    #    nixl-cu13, dragging a second cu13 torch + the whole cu13 nvidia stack
    #    (~1.5 GB of dead weight on a 550.x-driver / cu12.4 Pascal box). With --no-deps
    #    the resolver is bypassed, so install only the two cu12 pieces. The
    #    backend auto-selects nixl_cu12 (torch 12.4); NIXL bundles its own UCX
    #    (no separate ucx-py needed).
    uv pip install --python $V --no-deps nixl==1.4.0 nixl-cu12==1.4.0

Add **exactly one** `ld.so.conf.d` entry — for cufile, and only cufile.
torch 2.10 also `NEEDED`s `libcufile.so.0` (GPUDirect Storage), which the
Debian CUDA 12.4 toolkit does **not** ship; it comes from the venv's pip
`nvidia_cufile` package. Without this one entry `import torch` fails with
`ImportError: libcufile.so.0: cannot open shared object file`:

    printf '%s/venv/lib/python3.13/site-packages/nvidia/cufile/lib\n' $RT \
        > /etc/ld.so.conf.d/1cat-vllm-cufile.conf
    ldconfig

Do **not** add entries for the venv's other `nvidia/*/lib` dirs. torch uses
the **system** cu12 libs (above), not the cu13 `nvidia-*` pip packages that
the runtime deps pull in transitively (humming-kernels, flashinfer,
quack-kernels, tilelang). Those cu13 packages are present but are not the
active cuBLAS/cuDNN path — confs pointing at them are dead weight. The live
llama.lan runtime has exactly the one cufile conf and nothing else.

    # smoke test (verified against the dev0/dev5 wheels):
    $RT/venv/bin/python -c "
    import torch, vllm
    from vllm.benchmarks.lib.utils import default_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backend import AttentionType
    print(vllm.__version__, torch.version.cuda,
          current_platform.get_device_capability())
    with default_vllm_config():
        cfg = AttentionSelectorConfig(head_size=128, dtype=torch.float16,
            kv_cache_dtype='float16', block_size=16, use_mla=False, has_sink=False,
            use_sparse=False, use_mm_prefix=False, use_per_head_quant_scales=False,
            attn_type=AttentionType.DECODER, use_non_causal=False,
            use_batch_invariant=False, use_kv_connector=False)
        print(current_platform.get_attn_backend_cls(None, cfg, num_heads=24))
    "   # expect TRITON_ATTN on Pascal


### Step 3 — serve (canonical launch)

```bash
# CUDA_DEVICE_ORDER=PCI_BUS_ID is REQUIRED on this LXC: the default CUDA device
# ordering is NOT PCI-bus order, so without it CUDA_VISIBLE_DEVICES=0,1 selects
# the wrong physical GPUs.
# TRITON_CACHE_DIR pins the JIT cache. It is LOCAL — a fresh LXC starts
# cold (first boot ~5-15 min of Triton JIT + CUDA-graph capture) and is fast
# after that. Pin it to a durable path (not /tmp): /root/.triton-cache. The
# benchmark numbers in this doc are steady-state, so they hold cold or warm.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
TRITON_CACHE_DIR=/root/.triton-cache \
$RT/venv/bin/vllm serve /AI/models/Qwen3.8-27B-INT8-W8A16-MTP \
  --served-model-name qwen3.8-27b-int8 \
  --quantization compressed-tensors \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 32768 --max-num-seqs 8 \
  --kv-cache-dtype int8_per_token_head \
  --gpu-memory-utilization 0.92 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --host 0.0.0.0 --port 8905
# NOTE: Pascal (sm_60/61) has NO FP8, so the KV cache is int8_per_token_head
# (per-token-head scales; ~2x KV pool vs fp16). Fork builds with the fp8 alias (this build)
# also accept `fp8` as an alias for int8_per_token_head, but the older dev2
# build does NOT have that alias — pass int8_per_token_head explicitly so this
# block works on any build of the fork. Do NOT pass --enforce-eager.
```

Launchers for the tested configs live in `benchmarks/pascal/`
(`launch-fp8-mtp.sh`, `launch-awq-mtp.sh`, `launch-awq-mtp-cg.sh`).

## Operational notes (learned the hard way)

- **Cold boot is slow, not broken.** Fresh Triton cache + CUDA-graph capture =
  ~5–15 min on this 8-thread LXC (27B: `init engine ... took 808 s`). During
  that window the EngineCore logs `No available shared memory broadcast block
  found in 60 seconds` every minute and workers sit at 0% GPU — a **normal**
  phase. Do not kill the server for it; the port binds after "startup complete".
- **Kill by process group, never by parent PID.** The `vllm serve` parent and the
  `VLLM::EngineCore`/worker tree are separate; killing only the parent orphans the
  engine (it keeps loading, holds all the VRAM, and its port never comes up).
- **Stale `/dev/shm/psm_*`** from killed servers starve the next boot's broadcast
  pool — `rm -f /dev/shm/psm_*` only when the box is otherwise clean.
- **Never benchmark two servers at once** on this box (shared CPU + PCIe/SHM).
- **DVFS:** idle P40s sit at P8 (~1.2 GHz/715 MHz); the first bench after boot can
  read ~12% low. Warmup requests absorb this.
- Tool calls: the fork has **no plain `qwen` parser**; Qwen3.x instruct =
  `--tool-call-parser qwen3_xml` (with `hermes` you get HTTP 200 but the raw
  markup in `content` and empty `tool_calls` — a structurally wrong answer).

## Branches

| branch | state |
|---|---|
| `pascal-dev` | **build/deploy branch** — the consolidated working code (pre-upstream-merge Pascal fork + 12 hand-tuned kernel files + 5 custom GEMV kernels as committed `.so` + vendored `triton_kernels` + fp8→int8 KV alias + the CMake arch-whitelist / W8A8-fallback fixes). Clone **this** to build. This is what runs in prod. |
| `pascal` | **frozen** at f9d2ec4ba — the pre-consolidation prod tree, kept for reference only. Do not build from it. |

The experimental v1.3.0 upstream merge (which added the Exllama-v2 kernel set and
399 SM70 files) is **not** in this branch. Its tip is preserved as tag
`legacy/a561ea8a1-preconsolidate` if you ever want to diff against it.

Upstream: `origin` → 1CatAI/1Cat-vLLM (read-only for us; push 403).
This repo is a fork of it.

## NIXL KV-transfer pair (Qwen3.5-9B-w4a16, 4x P40)

Working 2026-08-31: two NixlConnector engines (`kv_role=kv_both`), each
TP=2 on a P40 pair, 16k context, `int8_per_token_head` 8-bit KV cache.
Launch: `bash scripts/vllm-nixl-launch.sh`.

Three non-obvious requirements (full rationale in the script header):

1. **`VLLM_SSM_CONV_STATE_LAYOUT=DS`** — GDN (Qwen3.5 Mamba-conv) +
   NixlConnector asserts this at worker init; the 3-read conv transfer
   needs dim-first layout. Default `SD` = instant `AssertionError`.
2. **`--compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'`** — no
   Dynamo; torch 2.10 hard-raises on the GDN layernorm Triton launch
   (`_cuda_getCurrentRawStream` returns int) when the default `-O2`
   VLLM_COMPILE mode traces it. GDN auto-downgrades to FULL_DECODE_ONLY.
3. **Distinct `VLLM_NIXL_SIDE_CHANNEL_PORT` per engine** (5600/5601) —
   both engines are `data_parallel_index 0`; the default port collides.

Model is **w4a16** (ExllamaLinearKernel, 5.25 GiB/P40 at TP=2), not
FP8-dynamic (17.73 GiB fp16-materialized) — the latter + 1 GB NIXL buffer
+ KV cache exceeds 24 GB. Observed per-P40 at TP=2: weights 5.25 GiB +
graphs 0.63 GiB + int8 KV 14.2 GiB (1.5M tokens) + NIXL 1 GB ~= 21-22.5 GiB.

