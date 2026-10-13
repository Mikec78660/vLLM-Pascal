# Pascal port: SM70 diagnostics channel disabled.
class _Channel:
    def __init__(self, name):
        self.name = name
        self.policy = type("_P", (object,), {"directory": False, "enabled": False})()


def diagnostic_channel(name):
    return _Channel(name)


__all__ = ["diagnostic_channel"]
