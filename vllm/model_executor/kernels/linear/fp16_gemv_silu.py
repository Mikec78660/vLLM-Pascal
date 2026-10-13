# Pascal port: SM70 FP16 GEMV+SiLU disabled.
class Sm70Fp16GemvSiluKernel:
    @classmethod
    def is_supported(cls, *a, **k):
        return (False, "SM70-only kernel")

    @classmethod
    def can_implement(cls, *a, **k):
        return (False, "SM70-only kernel")

    def process_weights_after_loading(self, layer):
        pass

    def apply_weights(self, *a, **k):
        raise RuntimeError("Sm70Fp16GemvSiluKernel is SM70-only; unavailable on Pascal")


__all__ = ["Sm70Fp16GemvSiluKernel"]
