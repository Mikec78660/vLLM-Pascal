# Pascal port: the SM70 execution-policy subsystem is absent. Provide a
# disabled policy so qwen4_exp's SM70 fast-paths are cleanly skipped.
class _DisabledPolicy:
    def __getattr__(self, name):
        return False
    def value(self, name, default=False):
        return default
    def __call__(self, *a, **k):
        return False

_disabled = _DisabledPolicy()


def layer_policy(config=None):
    return _disabled


def graph_policy(config=None):
    return _disabled


def ple_policy(config=None):
    return _disabled


def communication_policy(config=None):
    return _disabled


def capture_execution_policy(name, cls, config=None):
    return _disabled


def ple_cascade_configured():
    return False


class PlePlacementPolicy:
    """Env-var aliases for the PLE placement policy fields.

    The Pascal fork has no SM70 execution-policy subsystem, so the
    policy is read from the VLLM_QWEN4EXP_PLE_* environment variables.
    """
    aliases = {
        "host_gib": "VLLM_QWEN4EXP_PLE_HOST_GIB",
        "host_reserve_gib": "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB",
        "vram_reserve_gib": "VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB",
    }

    def __getattr__(self, name):
        return False


__all__ = [
    "PlePlacementPolicy",
    "layer_policy",
    "graph_policy",
    "ple_policy",
    "communication_policy",
    "capture_execution_policy",
    "ple_cascade_configured",
]
