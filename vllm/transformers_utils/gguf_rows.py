# Pascal port: SM70 GGUF packed-row reader disabled (NVFP4 safetensors path).
class PackedGGUFRowReader:
    def __init__(self, *a, **k):
        self._args = (a, k)

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None


__all__ = ["PackedGGUFRowReader"]
