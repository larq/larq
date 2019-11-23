from contextlib import contextmanager

__all__ = ["scope", "should_quantize"]

# TODO: Does this need to be a thread local variable
_SHOULD_QUANTIZE = False


@contextmanager
def scope(should_quantize):
    global _SHOULD_QUANTIZE
    backup = _SHOULD_QUANTIZE
    _SHOULD_QUANTIZE = should_quantize
    yield should_quantize
    _SHOULD_QUANTIZE = backup


def should_quantize():
    return _SHOULD_QUANTIZE
