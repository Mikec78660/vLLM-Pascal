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
    # Pascal port: the SM70 mapped-row-gather kernel is disabled,
    # so there is no GPU reader. Return (None, admission) so both
    # callers unpack a pair; the None reader routes the gather to
    # the numpy mmap fallback.
    return None, "pascal: no SM70 row kernel; numpy mmap gather"


__all__ = ["MappedRowGatherKernel", "prepare_mapped_row_gather"]
