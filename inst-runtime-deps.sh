#!/usr/bin/env bash
# Install the remaining runtime deps the smoke test needs (build does not need them).
# NOTE: excludes lru-disk (does not exist on PyPI; its presence aborted the earlier install).
cd /opt/1Cat-vLLM
/usr/local/bin/uv pip install --python /opt/vllm-build-env/bin/python \
    pillow triton compressed-tensors torchvision 2>&1 | tail -8
echo "=== verify ==="
/opt/vllm-build-env/bin/python -c "
import importlib.util as u
for m in ['pillow','triton','compressed_tensors','torchvision']:
    print(f'{m:18} {(\"OK\" if u.find_spec(m) else \"MISSING\")}')
import torch,torchvision; print('torchvision',torchvision.__version__,'| torch',torch.__version__)
import triton; print('triton',triton.__version__)
" 2>&1 | grep -viE "Detected|WARNING|FutureWarning" | tail -8
echo "RUNTIME_DEPS_DONE rc=$?"
