import tensorflow as _tf


@_tf.custom_gradient
def sign(x):
    def grad(dy):
        return dy

    return _tf.sign(x), grad


def binarize(x):
    return sign(_tf.clip_by_value(x, -1, 1))
