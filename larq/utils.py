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
