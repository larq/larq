import tensorflow as tf


@tf.custom_gradient
def sign(x):
    def grad(dy):
        return dy

    return tf.sign(x), grad


def binarize(x):
    return sign(tf.clip_by_value(x, -1, 1))
