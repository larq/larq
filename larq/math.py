"""Math operations that are specific to extremely quantized networks."""

import tensorflow as tf


def sign(x):
    r"""A sign function that will never be zero
    \\[
    f(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
    \end{cases}
    \\]

    This function is similar to
    [`tf.math.sign`](https://www.tensorflow.org/api_docs/python/tf/math/sign) but will
    return a binary value and will never be zero.

    # Arguments
    `x`: Input Tensor

    # Returns
    A Tensor with same type as `x`.
    """
    return tf.sign(tf.sign(x) + 0.1)
