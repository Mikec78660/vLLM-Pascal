# Pascal port: PLE offload disabled. The PLE ngram table (51 GB in FP8) is
# too large for 4xP40 VRAM, so on Pascal it is kept CPU-resident and gathered
# per token; the dedicated SM70 host/disk offload machinery is not used.
import torch
from torch import nn


def is_offload_process():
    return False


def ple_offload_enabled():
    return False


class PleOffloadLayer(nn.Module):
    """Minimal base for PLE embedding layers; offload is disabled on Pascal."""

    def __init__(self, *a, **k):
        super().__init__()


__all__ = ["PleOffloadLayer", "is_offload_process", "ple_offload_enabled"]
