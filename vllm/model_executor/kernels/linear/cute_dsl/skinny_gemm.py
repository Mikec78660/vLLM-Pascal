# Pascal port: SM70 CuteDSL skinny GEMM disabled.
class SkinnyGemmConfig:
    def __init__(self, *a, **k):
        self.raw = (a, k)


def shape_dynamic_skinny_gemm(*a, **k):
    return None


__all__ = ["SkinnyGemmConfig", "shape_dynamic_skinny_gemm"]
