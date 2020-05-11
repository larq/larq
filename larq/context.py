"""Context managers that configure global behaviour of Larq."""

import contextlib
import threading

__all__ = [
    "metrics_scope",
    "quantized_scope",
    "get_training_metrics",
    "should_quantize",
]


_quantized_scope = threading.local()
_quantized_scope.should_quantize = False


@contextlib.contextmanager
def quantized_scope(quantize):
    """A context manager to define the behaviour of `QuantizedVariable`.

    !!! example
        ```python
        model.save("full_precision_model.h5")  # save full precision latent weights
        fp_weights = model.get_weights()  # get latent weights

        with larq.context.quantized_scope(True):
            model.save("binary_model.h5")  # save binarized weights
            weights = model.get_weights()  # get binarized weights
        ```

    # Arguments
        quantize: If `should_quantize` is `True`, `QuantizedVariable` will return their
            quantized value in the forward pass. If `False`, `QuantizedVariable` will
            act as a latent variable.
    """
    backup = should_quantize()
    _quantized_scope.should_quantize = quantize
    yield quantize
    _quantized_scope.should_quantize = backup


def should_quantize():
    """Returns the current quantized scope."""
    return getattr(_quantized_scope, "should_quantize", False)


_global_training_metrics = set()
_available_metrics = {"flip_ratio"}


@contextlib.contextmanager
def metrics_scope(metrics=[]):
    """A context manager to set the training metrics to be used in quantizers.

    !!! example
        ```python
        with larq.context.metrics_scope(["flip_ratio"]):
            model = tf.keras.models.Sequential(
                [larq.layers.QuantDense(3, kernel_quantizer="ste_sign", input_shape=(32,))]
            )
        model.compile(loss="mse", optimizer="sgd")
        ```

    # Arguments
        metrics: Iterable of metrics to add to quantizers defined inside this context.
            Currently only the `flip_ratio` metric is available.
    """
    for metric in metrics:
        if metric not in _available_metrics:
            raise ValueError(
                f"Unknown training metric '{metric}'. Available metrics: {_available_metrics}."
            )
    backup = _global_training_metrics.copy()
    _global_training_metrics.update(metrics)
    yield _global_training_metrics
    _global_training_metrics.clear()
    _global_training_metrics.update(backup)


def get_training_metrics():
    """Retrieves a live reference to the training metrics in the current scope.

    Updating and clearing training metrics using `larq.context.metrics_scope` is
    preferred, but `get_training_metrics` can be used to directly access them.

    !!! example
        ```python
        get_training_metrics().clear()
        get_training_metrics().add("flip_ratio")
        ```

    # Returns
        A set of training metrics in the current scope.
    """
    return _global_training_metrics
