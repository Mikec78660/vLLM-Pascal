# Pascal port: SM70 PLE ngram kernel disabled.
class _Ngram:
    def reason(self, *a, **k):
        return "SM70 PLE ngram kernel disabled on Pascal"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return False


SM70_PLE_NGRAM = _Ngram()


def sm70_ple_ngram_ids(*a, **k):
    import torch
    return torch.empty(0, dtype=torch.long)


__all__ = ["SM70_PLE_NGRAM", "sm70_ple_ngram_ids"]
