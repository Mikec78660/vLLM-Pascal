#!/usr/bin/env bash
# 1Cat-vLLM NIXL pair — Qwen3.5-9B-w4a16, 16k ctx, int8 KV, 8-bit KV cache.
#   engine-8901: TP=2 on GPUs 0+1, port 8901, NIXL side-channel 5600
#   engine-8902: TP=2 on GPUs 2+3, port 8902, NIXL side-channel 5601
#
# Proven on 4x P40 (24 GB) 2026-08-31. Required elements, in order:
#
# 1. VLLM_SSM_CONV_STATE_LAYOUT=DS — REQUIRED for NixlConnector on GDN/Mamba
#    models. The worker asserts it (worker.py: "3-read Mamba conv transfer
#    requires DS conv state layout"). DS = (dim, state_len), dim on dim-1,
#    TP-consistent (mamba_utils.py); the qwen_gdn kernels have explicit DS
#    branches. Default is SD -> instant AssertionError at NIXL worker init.
#
# 2. --compilation-config '{"mode":0,"cudagraph_mode":"FULL"}' — README
#    canonical config. Mode 0 = no torch.compile/Dynamo, so the GDN
#    output-proj layernorm Triton launch is never traced and torch 2.10's
#    "torch.* op returned non-Tensor" (_cuda_getCurrentRawStream) crash
#    cannot fire. The GDN backend then auto-downgrades FULL ->
#    FULL_DECODE_ONLY (decode graphed, prefill eager). This is NOT
#    --enforce-eager; CUDA graphs are still captured.
#    Omitting this flag defaults to -O2 -> VLLM_COMPILE + FULL_AND_PIECEWISE
#    = Dynamo tracing = the crash.
#
# 3. Model: Qwen3.5-9B-w4a16 (int4 packed, group 128) — 5.25 GiB per P40 at
#    TP=2. The FP8-dynamic variant materializes to 17.73 GiB (fp16 dequant
#    fallback) which + 1 GB NIXL buffer + int8 KV does not fit 24 GB.
#    WNA16 path on Pascal = ExllamaLinearKernel (min-cap 60; committed).
#
# 4. --kv-cache-dtype int8_per_token_head — the fork's canonical 8-bit KV
#    (triton_attn uint8 kernels; committed source). NixlConnector sets KV
#    layout HND for xfer perf; the SSM conv state stays DS per (1).
#
# 5. Distinct VLLM_NIXL_SIDE_CHANNEL_PORT per engine (5600/5601): both
#    engines are data_parallel_index 0, so the default 5600 collides.
#
# Memory at TP=2 per P40: weights 5.25 + graphs ~0.63 + KV cache ~14.2 GiB
# (1.5M tokens, ~28x 16k concurrency) + 1 GB NIXL buffer ~= 21-22.5 GiB,
# fits 24 GB at gpu-mem-util 0.92.
set -u
V=/mnt/AI/1Cat-vLLM/venv/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRITON_CACHE_DIR=/root/.triton-cache
export VLLM_SSM_CONV_STATE_LAYOUT=DS

mkdir -p /root/vllm-nixl
: > /root/vllm-nixl/pids.txt

launch() {
  local gpus="$1" port="$2" scport="$3" eid="$4"
  echo "launching $eid on GPUs $gpus (TP=2), port $port, side-channel $scport"
  CUDA_VISIBLE_DEVICES="$gpus" \
  VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
  VLLM_NIXL_SIDE_CHANNEL_PORT="$scport" \
  nohup "$V/vllm" serve /AI/models/Qwen3.5-9B-w4a16 \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --dtype float16 \
    --kv-cache-dtype int8_per_token_head \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL"}' \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"engine_id\":\"$eid\"}" \
    > "/root/vllm-nixl/server-$port.log" 2>&1 &
  echo "$! $port $eid" >> /root/vllm-nixl/pids.txt
}

launch 0,1 8901 5600 engine-8901
launch 2,3 8902 5601 engine-8902
echo "launched; pids:"
cat /root/vllm-nixl/pids.txt
