# Pascal port: GDN projection policy disabled.
class _Disabled:
    def __getattr__(self, name):
        return False
    def value(self, name, default=False):
        return default

_disabled = _Disabled()


def projection_policy(config=None):
    return _disabled


__all__ = ["projection_policy"]
