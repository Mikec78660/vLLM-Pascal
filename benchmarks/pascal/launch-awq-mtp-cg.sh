#!/usr/bin/env bash
# MTP + CUDA-GRAPHS variant (eager model, FULL cudagraphs).
#
# Why this exists: --enforce-eager was a sledgehammer that also disabled
# cudagraphs. Root cause of the earlier graphs crash (torch 2.10 + triton 3.6):
# with the default CompilationMode (VLLM_COMPILE), dynamo traces the whole
# Qwen3NextModel.forward fullgraph; inside it the Triton W4A16 GEMM launcher
# calls driver.get_current_stream() -> torch._C._cuda_getCurrentRawStream,
# which returns an int. Dynamo cannot trace an int-returning builtin into an
# FX graph -> hard fail ("torch.* op returned non-Tensor").
#
# Fix: compilation mode 0 (NONE) = NO inductor, NO dynamo, model runs eager;
# but CUDAGraphWrapper still captures the eager forward into CUDA graphs
# (cudagraph_mode=FULL). TRITON_ATTN + GDN both declare AttentionCGSupport
# >= UNIFORM_BATCH, which is what FULL decode graphs need with spec-decode.
# This is the vLLM-blessed "eager + full cudagraph" combination.
#
# Devices: CUDA 2,3 = P40 (cap 6.1). CUDA 0,1 = P100 (do NOT use; llama runs
# there). Override with CUDA_VISIBLE_DEVICES=... if needed.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"  # PCI_BUS_ID order: 0,1 = free P40s
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export NCCL_P2P_DISABLE=1

VENV=/mnt/AI/1Cat-vLLM/venv
MODEL=/AI/models/Qwen3.8-27B-AWQ-INT4
PORT="${PORT:-8901}"

exec $VENV/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b-awq-int4-mtp \
  --quantization compressed-tensors \
  --language-model-only \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL\"}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2,\"enforce_eager\":true,\"use_local_argmax_reduction\":true}" \
  --host 0.0.0.0 \
  --port "$PORT"
