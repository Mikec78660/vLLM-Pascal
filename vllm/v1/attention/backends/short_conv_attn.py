# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class ShortConvAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:
        return ShortConvAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    metadata_cls = ShortConvAttentionMetadata

# Qwen4Exp port: PLE short-conv attention backend. On Pascal the PLE ngram
# table is CPU-resident; the short-conv state path reuses the standard
# ShortConv backend with PLE's metadata shape.
class PleShortConvAttentionMetadata(ShortConvAttentionMetadata):
    """PLE short-conv metadata (aliases the standard short-conv metadata)."""


class PleShortConvAttentionBackend(ShortConvAttentionBackend):
    """PLE short-conv backend (reuses the standard short-conv backend)."""

    @classmethod
    def get_name(cls) -> str:
        return "PLE_SHORT_CONV"
