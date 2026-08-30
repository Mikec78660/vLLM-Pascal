#!/bin/bash
# Qwen3.8-27B-FP8 (blockwise [128,128] e4m3, unsloth/Qwen mirror of Qwen official)
# on 4x P40 TP=4 + MTP + CUDA graphs (mode=0 eager model, FULL_DECODE_ONLY).
#
# FP8 on Pascal: native fp8 gemm needs sm_89; this fork's dequant-fallback
# converts block weights to fp16 at load (~13.9 GiB/GPU at TP=4) and runs
# plain F.linear (cuBLAS fp16) — the fastest available path on CC 6.x.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"  # PCI order: 4 free P40s
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export NCCL_P2P_DISABLE=0
# export VLLM_USE_NCCL_SYMM_MEM=1  # needs NCCL>=2.27; host has 2.22.3 -> disabled
export VLLM_NCCL_INCLUDE_PATH=/tmp/nccl-hdr

VENV=/mnt/AI/1Cat-vLLM/venv
MODEL=/AI/models/Qwen3.8-27B-FP8
PORT="${PORT:-8902}"

exec $VENV/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b-fp8-mtp \
  --quantization fp8 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-model-len "${MAX_MODEL_LEN:-90112}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL\"}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2,\"enforce_eager\":true,\"use_local_argmax_reduction\":true}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${EXTRA_ARGS:+$EXTRA_ARGS}
