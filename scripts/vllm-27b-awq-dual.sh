#!/usr/bin/env bash
# Qwen3.8-27B-AWQ-INT4 dual deployment on llama.lan (2026-08-31):
#   8901: P100s 5,6 (TP=2), 1 session, ctx 190000
#   8902: P40s 0,1,2,3 (TP=4), 6 sessions, ctx 190000
# NOTE: the P100 pair needs --gpu-memory-utilization 0.96 (override: UTIL=0.96) - at 0.92 its KV budget (1.48 GiB) fails the hard KV check (needs 3.03 GiB for 190k); 0.96 yields a 218,650-token pool = 1.15x, the maximum that fits 16 GB cards. The P40s run 0.92 comfortably (1.84M-token pool, 9.66x).
Both: fp8 KV (Pascal-aliased -> int8_per_token_head), mode 0 + FULL
# (no Dynamo; torch 2.10 safe), VLLM_SSM_CONV_STATE_LAYOUT=DS.
set -u
V=/mnt/AI/1Cat-vLLM/venv/bin
MODEL=/AI/models/Qwen3.8-27B-AWQ-INT4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRITON_CACHE_DIR=/root/.triton-cache
export VLLM_SSM_CONV_STATE_LAYOUT=DS
mkdir -p /root/vllm-nixl
: > /root/vllm-nixl/pids.txt

serve() {
  local gpus="$1" tp="$2" port="$3" sess="$4"
  echo "launching port $port | GPUs $gpus | TP=$tp | sessions=$sess | ctx=190000"
  CUDA_VISIBLE_DEVICES="$gpus" \
  nohup "$V/vllm" serve "$MODEL" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size "$tp" \
    --max-model-len 190000 \
    --max-num-seqs "$sess" \
    --gpu-memory-utilization 0.92 \
    --dtype float16 \
    --kv-cache-dtype fp8 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL"}' \
    > "/root/vllm-nixl/server-$port.log" 2>&1 &
  echo "$! $port" >> /root/vllm-nixl/pids.txt
}

serve 5,6      2 8901 1
serve 0,1,2,3  4 8902 6
echo "launched:"
cat /root/vllm-nixl/pids.txt
