from tensorflow import __version__
from tensorflow.keras.utils import get_custom_objects


def get_tf_version_major_minor_float():
    TF_VERSION_MAJOR, TF_VERSION_MINOR, TF_VERSION_PATCH = __version__.split(".", 2)
    return float(TF_VERSION_MAJOR + "." + TF_VERSION_MINOR)


def register_keras_custom_object(cls):
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    get_custom_objects()[cls.__name__] = cls
    return cls
