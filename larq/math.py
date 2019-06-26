"""Math operations that specific to extremely quantized networks."""

import tensorflow as tf


def sign(x):
    r"""A sign function that will never be zero
    \\[
    f(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
    \end{cases}
    \\]

    # Arguments
    `x`: Input `tensor`

    # Returns
    A `tensor` with same type as `x`.
    """
    return tf.sign(tf.sign(x) + 0.1)
