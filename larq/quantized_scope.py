import contextlib
import threading

__all__ = ["scope", "should_quantize"]

_quantized_scope = threading.local()
_quantized_scope.should_quantize = False


@contextlib.contextmanager
def scope(should_quantize):
    """A context manager to define the behaviour of `QuantizedVariable`.

    # Arguments
    should_quantize: If `should_quantize` is `True`, `QuantizedVariable` will return
        their quantized value in the forward pass. If `False`, `QuantizedVariable`
        will act as a latent variable.
    """
    backup = _quantized_scope.should_quantize
    _quantized_scope.should_quantize = should_quantize
    yield should_quantize
    _quantized_scope.should_quantize = backup


def should_quantize():
    """Returns the current quantized scope."""
    return _quantized_scope.should_quantize
