# Pascal port: SM70 attention workspaces disabled.
def workspace_cache(*a, **k):
    return None


def retain_for_capture(*a, **k):
    return None


__all__ = ["workspace_cache", "retain_for_capture"]
