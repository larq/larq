from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects


def register_keras_custom_object(cls):
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    get_custom_objects()[cls.__name__] = cls
    return cls


def tf_1_14_or_newer():
    return LooseVersion(tf.__version__) >= LooseVersion("1.14.0")
