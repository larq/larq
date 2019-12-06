import contextlib
import threading

_quantized_scope = threading.local()
_quantized_scope.should_quantize = False


@contextlib.contextmanager
def scope(quantize):
    """A context manager to define the behaviour of `QuantizedVariable`.

    # Arguments
    quantize: If `should_quantize` is `True`, `QuantizedVariable` will return their
        quantized value in the forward pass. If `False`, `QuantizedVariable` will act
        as a latent variable.
    """
    backup = should_quantize()
    _quantized_scope.should_quantize = quantize
    yield quantize
    _quantized_scope.should_quantize = backup


def should_quantize():
    """Returns the current quantized scope."""
    return getattr(_quantized_scope, "should_quantize", False)
