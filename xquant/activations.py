import tensorflow as tf
from xquant import utils


@utils.register_keras_custom_object
def hard_tanh(x):
    r"""Hard tanh activation function.
    \\[\sigma(x) = \mathrm{Clip}(x, −1, 1)\\]

    # Arguments
    x: Input tensor.

    # Returns
    Hard tanh activation.
    """
    return tf.clip_by_value(x, -1, 1)
