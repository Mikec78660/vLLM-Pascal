"""HF config re-export for Qwen4Exp (flash-next) models.
The real config class lives in vllm.models.qwen4_exp.config so that
vLLM config loading resolves to our ported implementation.
"""
from vllm.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig

__all__ = ["Qwen4ExpConfig", "Qwen4ExpTextConfig"]
