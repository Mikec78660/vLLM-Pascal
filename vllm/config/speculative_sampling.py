# Pascal port: SM70 speculative-sampling (batched MTP) policy disabled.
def mtp_batch_enabled(config=None):
    return False


__all__ = ["mtp_batch_enabled"]
