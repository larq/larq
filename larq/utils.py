from contextlib import contextmanager

from tensorflow.keras.utils import get_custom_objects


def memory_as_readable_str(num_bits: int) -> str:
    """Generate a human-readable string for the memory size.

    1 KiB = 1024 B; we use the binary prefix (KiB) [1,2] instead of the decimal prefix
    (KB) to avoid any confusion with multiplying by 1000 instead of 1024.

    [1] https://en.wikipedia.org/wiki/Binary_prefix
    [2] https://physics.nist.gov/cuu/Units/binary.html
    """

    suffixes = ["B", "KiB", "MiB", "GiB"]
    num_bytes = num_bits / 8

    for i, suffix in enumerate(suffixes):
        rounded = num_bytes / (1024 ** i)
        if rounded < 1024:
            break

    return f"{rounded:,.2f} {suffix}"


def register_keras_custom_object(cls):
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    get_custom_objects()[cls.__name__] = cls
    return cls


def register_alias(name: str):
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


def set_precision(precision: int = 32):
    """A decorator to set the precision of a quantizer function

    # Arguments
        precision: An integer defining the precision of the output.
    """

    def decorator(function):
        setattr(function, "precision", precision)
        return function

    return decorator


@contextmanager
def patch_object(object, name, value):
    """Temporarily overwrite attribute on object"""
    old_value = getattr(object, name)
    setattr(object, name, value)
    yield
    setattr(object, name, old_value)
