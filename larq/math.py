"""Math operations that specific to extremely quantized networks."""

import tensorflow as tf


def sign(x):
    """A sign function that will never be zero"""
    return tf.sign(tf.sign(x) + 0.1)
