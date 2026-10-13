# Pascal port: SM70 PLE GGUF pinned-table affinity disabled.
def pinned_decode_active(*a, **k):
    return False


def pinned_table_load_affinity(*a, **k):
    return False


__all__ = ["pinned_decode_active", "pinned_table_load_affinity"]
