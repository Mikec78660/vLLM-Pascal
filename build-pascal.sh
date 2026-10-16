#!/usr/bin/env bash
set -o pipefail
cd /opt/1Cat-vLLM || exit 1
export PATH=/opt/vllm-build-env/bin:$PATH   # cmake 4.4.3 before /usr/bin/cmake 3.25 (>=3.26 required)
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="6.0;6.1"
export VLLM_SKIP_SM70_BUNDLES=1
export VLLM_REQUIRE_RUST_FRONTEND=0
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS=4      # 27 GB box: notes say 8 OOMs; 4 + NVCC_THREADS=1 is the safe floor
export NVCC_THREADS=1
echo "[build] start $(date) arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS commit=$(git log --oneline -1)"
/opt/vllm-build-env/bin/python setup.py bdist_wheel 2>&1
rc=$?
echo "[build] done $(date) exit=$rc"
ls -la dist/ 2>/dev/null
exit $rc
