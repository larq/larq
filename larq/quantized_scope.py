import contextlib
import threading

__all__ = ["scope", "should_quantize"]

_quantized_scope = threading.local()
_quantized_scope.should_quantize = False


@contextlib.contextmanager
def scope(should_quantize):
    backup = _quantized_scope.should_quantize
    _quantized_scope.should_quantize = should_quantize
    yield should_quantize
    _quantized_scope.should_quantize = backup


def should_quantize():
    return _quantized_scope.should_quantize
