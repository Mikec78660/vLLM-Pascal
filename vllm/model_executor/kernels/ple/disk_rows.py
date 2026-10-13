# Pascal port: SM70 PLE mapped-row-gather kernel disabled.
class MappedRowGatherKernel:
    def __init__(self, *a, **k):
        self._args = (a, k)

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def prepare_mapped_row_gather(*a, **k):
    return None


__all__ = ["MappedRowGatherKernel", "prepare_mapped_row_gather"]
