# Pascal port: SM70 decode-graph compilation is disabled.
from contextlib import contextmanager


def is_sm70_decode_graph_compiling():
    return False


def use_sm70_decode_graph_semantics(policy=None):
    return True


@contextmanager
def sm70_decode_graph_compilation(enabled=True):
    yield None


__all__ = [
    "is_sm70_decode_graph_compiling",
    "use_sm70_decode_graph_semantics",
    "sm70_decode_graph_compilation",
]
