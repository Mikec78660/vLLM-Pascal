# SPDX-License-Identifier: Apache-2.0
"""Pascal (CC 6.1) Q8_0 GGUF-DP4A linear method.

Replaces the SM70 FP8 dequant-to-fp16 load-time fallback for 128x128
block-quantized checkpoints on Pascal GPUs.

Instead of materializing fp16 weights (2 bytes/weight; P40 fp16 has no
tensor-core path), the FP8 block weights are dequantized to fp32 ONCE at
load time and requantized to GGUF Q8_0 (1.0625 bytes/weight; per-32-elem
{half scale; int8[32]} blocks). Inference then runs the fork's __dp4a()
GGUF kernels (int8 x int8 -> int32 -> fp32 accumulation, llama.cpp
MMQ/MMVQ semantics) with the per-32-block scale applied in-kernel:

  M == 1  -> torch.ops._C.ggml_mul_mat_vec_a8  (decode, bandwidth-bound)
  M >= 2  -> torch.ops._C.ggml_mul_mat_a8      (prefill / small batch)

The per-32-elem scale is computed over the LOCAL TP shard, matching the
sharded weight actually resident on this rank.

Gate: envs.VLLM_PASCAL_FP8_Q80 (default "1"), effective only when the
device is Pascal (CC 6.x), the checkpoint is 128x128 block-quantized FP8,
and K (input_size_per_partition) is divisible by 32. Set the env var to
"0" to restore the historical fp16 dequant fallback exactly.
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8LinearMethod,
    _get_w16a16_gemv,
    _w16a16_apply,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

_GGUF_Q8_0 = 8  # GGMLQuantizationType.Q8_0


def _is_pascal_q80_enabled() -> bool:
    import vllm.envs as envs

    if not envs.VLLM_PASCAL_FP8_Q80:
        return False
    if not current_platform.is_cuda():
        return False
    cap = current_platform.get_device_capability()
    return cap is not None and cap[0] == 6


def q80_pack_blocks(
    weight_f32: torch.Tensor,
    block_k: int = 32,
    scale_dtype: torch.dtype = torch.float16,
) -> torch.Tensor | None:
    """Requantize an [N, K] fp32 (dequantized) weight to packed GGUF Q8_0.

    Returns uint8 tensor [N, K//block_k * 34]; each 32-elem block is laid
    out as [d: half (2 bytes)] [qs: int8 x 32]. Scale per 32-elem
    block over the LOCAL shard.

    SCALE CONVENTION (verified empirically against the deployed kernel on
    real checkpoint weights): the fork's Q8_0 kernel MULTIPLIES by the
    stored half — out = d * sum(x*qs) — which is exactly the standard
    GGML q8_0 convention (d = amax/127, a scale; dequant = qs*d). So pack
    d = amax/127 and qs = round(w/d). The ORIGINAL packer inverted this
    (d = 127/amax): with typical amax ~0.1-1.0 that d is huge, so
    qs = round(w/d) rounded every element to 0 and the kernel output
    collapsed to ~0. d is clamped to [1e-30, 65500] so it stays a finite
    fp16; qs saturates at +/-127 (precision loss only).

    NOTE: the GGUF Q8_0 kernel layout is ALWAYS 32-elem blocks — pass
    block_k=32 even when the source checkpoint uses 128-elem FP8 blocks.
    The 128-block scale is only used for the intermediate fp32 dequant;
    the per-32 scale is re-derived from the dequantized values.
    """
    N, K = weight_f32.shape
    if K % block_k != 0:
        return None
    w = weight_f32.reshape(N, K // block_k, block_k)
    amax = w.abs().amax(dim=2).clamp(min=1e-30, max=65500.0)
    d = (amax / 127.0)
    qs = (w / d.unsqueeze(2)).round().clamp(-127, 127).to(torch.int8)
    packed = torch.empty(N, K // block_k, 34, dtype=torch.uint8,
                         device=weight_f32.device)
    packed[:, :, :2] = (
        d.to(scale_dtype).contiguous().view(torch.uint8)
        .view(N, K // block_k, 2)
    )
    packed[:, :, 2:] = qs.view(N, K // block_k, block_k).view(torch.uint8)
    return packed.reshape(N, -1).contiguous()


class Q80LinearMethod(Fp8LinearMethod):
    """FP8 block-quant checkpoint -> GGUF Q8_0 DP4A inference on Pascal."""

    def __init__(self, quant_config):
        # Fp8LinearMethod.__init__ sets use_sm70_dequant_fallback based on
        # the envs; we force the fallback branch semantics (its
        # create_weights early-returns before kernel init, and its
        # process_weights_after_loading takes the dequant path) but then
        # replace the fp16 dequant with an in-place Q8_0 requantize.
        super().__init__(quant_config)
        self.use_sm70_dequant_fallback = True
        self.fp8_linear = None
        self.use_marlin = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert self.block_quant and self.weight_block_size is not None
        _, block_k = self.weight_block_size

        weight, scale_inv = process_fp8_weight_block_strategy(
            layer.weight, layer.weight_scale_inv
        )
        weight_f32 = self._dequantize_block_weight(
            weight, scale_inv, torch.float32
        )
        packed = q80_pack_blocks(
            weight_f32, 32, layer.orig_dtype
        )
        if packed is None:
            # K not a multiple of 32: historical fp16 dequant behavior
            # for this layer (identical to the old fallback).
            w16 = self._dequantize_block_weight(
                weight, scale_inv, layer.orig_dtype
            )
            if (
                w16.dtype == torch.float16
                and w16.dim() == 2
                and w16.size(0) % 8 == 0
                and _get_w16a16_gemv() is not None
            ):
                w16 = w16.t().contiguous()
                layer.sm70_w16_gemv = True
            replace_parameter(layer, "weight", w16)
            replace_parameter(layer, "weight_scale_inv", None)
            layer.input_scale = None
            logger.warning_once(
                "[PASCAL-Q80] %s: K=%d not divisible by 32; kept the fp16 "
                "dequant fallback for this layer.",
                getattr(layer, "prefix", "?"),
                layer.input_size_per_partition,
            )
            return

        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_scale_inv", None)
        layer.input_scale = None
        layer.q80_row = packed.shape[0]
        layer.q80_block_k = block_k
        logger.info(
            "[PASCAL-Q80] %s: FP8 128x128 weights requantized to GGUF "
            "Q8_0 in place (%d x %d, %d packed bytes = 1.0625 B/weight); "
            "inference via __dp4a MMQ/MMVQ, no fp16 weights.",
            getattr(layer, "prefix", "?"),
            packed.shape[0],
            layer.input_size_per_partition,
            packed.numel(),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(layer, "q80_row"):
            # Layer fell back to the fp16 dequant path (K % 32 != 0).
            w = layer.weight
            if getattr(layer, "sm70_w16_gemv", False) and x.dim() == 2:
                if x.shape[0] <= 4:
                    out = _w16a16_apply(x, w, bias)
                    if out is not None:
                        return out
                else:
                    out = torch.matmul(x, w)
                    if bias is not None:
                        out = out + bias
                    return out
            return torch.nn.functional.linear(x, w, bias)

        N = layer.q80_row
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        M = x_2d.shape[0]
        if M == 1:
            out = ops.ggml_mul_mat_vec_a8(layer.weight, x_2d, _GGUF_Q8_0, N)
        else:
            out = ops.ggml_mul_mat_a8(layer.weight, x_2d, _GGUF_Q8_0, N)
        out = out.view(*x.shape[:-1], N)
        if bias is not None:
            out = out + bias
        return out


__all__ = ["Q80LinearMethod", "q80_pack_blocks", "_is_pascal_q80_enabled"]
