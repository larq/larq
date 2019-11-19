from contextlib import contextmanager

from tensorflow.keras.utils import get_custom_objects


def register_keras_custom_object(cls):
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    get_custom_objects()[cls.__name__] = cls
    return cls


def register_alias(name):
    """A decorator to register a custom keras object under a given alias.
    !!! example
        ```python
        @utils.register_alias("degeneration")
        class Degeneration(tf.keras.metrics.Metric):
            pass
        ```
    """

    def register_func(cls):
        get_custom_objects()[name] = cls
        return cls

    return register_func


def set_precision(precision=32):
    """A decorator to set the precision of a quantizer function

    # Arguments
    precision: An integer defining the precision of the output.
    """

    def decorator(function):
        setattr(function, "precision", precision)
        return function

    return decorator


@contextmanager
def quantize(layer, kernel_name, quantizer):
    """Temporarily apply a quantizer to a kernel.

    This is needed, since we do not want to mutate existing references that might
    expect a tf.Variable instead of a tf.Tensor. This can happen in eager mode since
    overwriting a kernel would also mutate layer.trainable_variables which breaks
    gradient computation.
    """
    full_precision_kernel = getattr(layer, kernel_name)
    if quantizer is None:
        yield full_precision_kernel
    else:
        quantized_kernel = quantizer(full_precision_kernel)
        setattr(layer, kernel_name, quantized_kernel)
        yield quantized_kernel
        setattr(layer, kernel_name, full_precision_kernel)
