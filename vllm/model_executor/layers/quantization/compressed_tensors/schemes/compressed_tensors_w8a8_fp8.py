# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable

from vllm.platforms import current_platform

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    STRATEGY_TO_PARAMETER_TYPE,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)

__all__ = ["CompressedTensorsW8A8Fp8"]

STATIC_QUANT = True
DYNAMIC_QUANT = False
activation_quant_key_mapping = {
    STATIC_QUANT: kFp8StaticTensorSym,
    DYNAMIC_QUANT: kFp8DynamicTokenSym,
}
weight_quant_key_mapping = {
    QuantizationStrategy.CHANNEL: kFp8StaticChannelSym,
    QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
}
logger = init_logger(__name__)


_W8A16_GEMV = None


def _get_w8a16_gemv():
    global _W8A16_GEMV
    if _W8A16_GEMV is not None:
        return _W8A16_GEMV if _W8A16_GEMV is not False else None
    import os
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(
        here, "../../../../kernels/linear/mixed_precision/w8a16_gemv_v11.so"))
    try:
        spec = importlib.util.spec_from_file_location("w8a16_gemv_v11", cand)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _W8A16_GEMV = m
        return m
    except Exception as e:
        raise RuntimeError(f"w8 gemv load failed: {e}") from e


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure

        if self.weight_block_size is not None:
            self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
            self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enabled()
            assert not self.is_static_input_scheme
            self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
            self.activation_quant_key = create_fp8_quant_key(
                static=False, group_shape=self.act_q_group_shape
            )
        else:
            self.activation_quant_key = activation_quant_key_mapping[
                self.is_static_input_scheme
            ]
            self.weight_quant_key = weight_quant_key_mapping[self.strategy]

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up — EXCEPT Pascal (CC 6.x), which runs this scheme
        # through a load-time dequant->fp16 fallback (see
        # process_weights_after_loading). Kill switch: VLLM_CT_W8A8_PASCAL=0.
        if os.environ.get("VLLM_CT_W8A8_PASCAL", "1") == "0":
            return 89
        return 60

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = create_fp8_scale_parameter(
            STRATEGY_TO_PARAMETER_TYPE[self.strategy],
            output_partition_sizes,
            input_size_per_partition,
            layer.weight_block_size,
            weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            layer.register_parameter("input_scale", input_scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            weight_shape=(output_size_per_partition, input_size_per_partition),
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer) -> None:
        # --- Pascal (CC 6.x): no fp8 scaled-mm hardware. Dequantize fp8
        # weights (per-channel OR per-tensor scales) to fp16 at load and run
        # plain F.linear in apply_weights. Mirrors the SM70 fp8.py blockwise
        # dequant fallback. Kill switch: VLLM_CT_W8A8_PASCAL=0.
        _cap = None
        if current_platform.is_cuda():
            _ctup = current_platform.get_device_capability()
            if _ctup is not None:
                _cap = _ctup.to_int()
        if (
            os.environ.get("VLLM_CT_W8A8_PASCAL", "1") != "0"
            and _cap is not None and _cap < 70
        ):
            logger.info_once(
                "CT W8A8-FP8 Pascal dequant fallback active (capability %d): "
                "materializing fp16 weights.", _cap)
            w = layer.weight            # [N, K] fp8 e4m3, as loaded
            s = layer.weight_scale      # scalar | [N] | [N,1] (channel/tensor)
            if s.dim() == 0:
                s = s.reshape(1, 1)
            elif s.numel() == w.shape[0]:
                s = s.reshape(-1, 1)
            else:
                raise RuntimeError(
                    "CT W8A8 Pascal fallback: unsupported weight_scale shape "
                    f"{tuple(s.shape)} for weight {tuple(w.shape)}")
            # dual residency: fp16 [N,K] for prefill + any consumer;
            # int8 bytes [K,N] + fp16 scales [N] feed the decode GEMV
            # (half the DRAM traffic per generated token)
            w16 = (w.to(torch.float32) * s.to(torch.float32)).half()
            layer.weight = Parameter(w16.contiguous().data,
                                     requires_grad=False)
            w8b = w.view(torch.int8).t().contiguous()
            sc16 = s.reshape(-1).to(torch.float16).contiguous()
            layer.w8_bytes = Parameter(w8b.data, requires_grad=False)
            layer.w8_scales = Parameter(sc16.data, requires_grad=False)
            layer.input_scale = None
            self._pascal_dequant_fallback = True
            self._pascal_int8_direct = True
            mod = _get_w8a16_gemv()
            lut = torch.arange(256, dtype=torch.uint8).view(
                torch.float8_e4m3fn).to(torch.float32)
            mod.build_e4m3_lut(lut.cpu())
            return
        if self.strategy == QuantizationStrategy.TENSOR:
            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                layer.weight,
                layer.weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            weight = weight.t()
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                layer.weight, layer.weight_scale, getattr(layer, "input_scale", None)
            )
            weight = weight.t()

        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            self.fp8_linear.process_weights_after_loading(layer)

            layer.input_scale = None
            # fp8_linear.process_weights_after_loading applies the post process
            # and reassigns the weight and weight_scale buffers to layer attributes.
            return

        else:
            raise ValueError(
                f"Unknown quantization strategy {self.strategy}: "
                f"should be one of {list(QuantizationStrategy)}"
            )

        # required by torch.compile to be torch.nn.Parameter
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        if input_scale is not None:
            layer.input_scale = Parameter(input_scale.data, requires_grad=False)

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

        if hasattr(self, "fp8_linear"):
            self.fp8_linear.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_pascal_dequant_fallback", False):
            from torch.nn import functional as F
            unflatten = x.dim() > 2
            x2 = x.reshape(-1, x.shape[-1]) if unflatten else x
            use_gemv = (
                getattr(self, "_pascal_int8_direct", False)
                and x2.shape[0] <= 4 and x2.dtype == torch.float16
            )
            if use_gemv:
                mod = _get_w8a16_gemv()
                out = torch.empty(x2.shape[0], layer.w8_bytes.shape[1],
                                  dtype=torch.float16, device=x.device)
                mod.w8a16_gemv(layer.w8_bytes, layer.w8_scales,
                               x2.contiguous(), out, 0)
            else:
                out = F.linear(x2.half() if x2.dtype != torch.float16
                               else x2, layer.weight,
                               bias if bias is None else bias.to(x.dtype))
            if bias is not None:
                out = out + bias.to(out.dtype)
            if unflatten:
                out = out.reshape(*x.shape[:-1], out.shape[-1])
            return out.to(x.dtype) if out.dtype != x.dtype else out
        return self.fp8_linear.apply_weights(layer, x, bias)
