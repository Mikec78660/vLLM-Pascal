#!/usr/bin/env bash
# MTP variant of launch-awq.sh: enables Qwen3.5 MTP speculative decoding.
# Same NCCL/SHM + eager + INT4-CT base as launch-awq.sh, plus an explicit
# --speculative-config. (The fork's VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS auto-env
# is V100-gated [cap.major != 7: return], so it silently no-ops on our P40s —
# the explicit config below is the path that works on Pascal.)
#
# Why these speculative-config knobs:
#  method=mtp
#      Uses the checkpoint's native MTP head (text_config.mtp_num_hidden_layers=1)
#      as the draft model. The fork auto-rewires hf model_type qwen3_5 -> qwen3_5_mtp
#      and arch -> Qwen3_5MTP (registry entry present).
#  num_speculative_tokens=2
#      MUST be set explicitly: the rewire reads mtp_num_hidden_layers from the TOP-
#      level config, which lacks it (it lives in text_config), so n_predict resolves
#      to None and vLLM would raise "num_speculative_tokens was not provided"
#      otherwise. k=2 matches llama.cpp --spec-draft-n-max 2 for a fair A/B; the
#      single MTP layer is reused per draft token (divisibility check passes).
#  enforce_eager=true
#      Pascal (sm_61) cannot capture the drafter's NCCL allreduce in a CUDA graph
#      (cudaErrorStreamCaptureUnsupported on 4 PCIe P40s); keep the drafter eager.
#  use_local_argmax_reduction=true
#      Each rank does a local argmax and only all-reduces the small result, cutting
#      the SHM-transport allreduce traffic that is the dominant per-token cost here.
#
# Single-session mode: --max-model-len 32768 (model native cap is 262144;
# KV pool is GPU-side, headroom is fine) + --max-num-seqs 1 (hard cap: one
# generation at a time; extra requests queue). Override with MAX_MODEL_LEN /
# MAX_NUM_SEQS env if you ever need more.
#
# Quantization: the MTP draft linears load UNQUANTIZED (BF16). The checkpoint's
# CT ignore list covers mtp.* (mtp.fc, mtp.layers.0 mlp+attn), so get_scheme()
# returns None for them -> UnquantizedLinearMethod -> clean BF16 load. embed/lm_head
# are PPMissingLayer (shared with the target, no extra memory).
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
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
  --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"enforce_eager":true,"use_local_argmax_reduction":true}' \
  --host 0.0.0.0 \
  --port "$PORT"
