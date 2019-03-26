import tensorflow as tf
from xquant import utils


def sign(x):
    """A sign function that will never be zero"""
    return tf.sign(tf.sign(x) + 1e-10)


@utils.register_keras_custom_object
@tf.custom_gradient
def ste_sign(x):
    r"""
    Sign binarization function.
    \\[
    q(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
   \end{cases}
   \\]

    The gradient is estimated using the Straight-Through Estimator.
    \\[\frac{\partial q(x)}{\partial x} = x\\]

    # Arguments
    x: Input tensor.

    # Returns
    Binarized tensor.

    # References
    - [Binarized Neural Networks: Training Deep Neural Networks with Weights and
      Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)
    """

    def grad(dy):
        return dy

    return sign(x), grad


@utils.register_keras_custom_object
@tf.custom_gradient
def approx_sign(x):
    r"""
    Sign binarization function.
    \\[
    q(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
   \end{cases}
   \\]

    The gradient is estimated using the ApproxSign method.
    \\[\frac{\partial q(x)}{\partial x} = (2 - 2 \left|x\right|))\\]

    # Arguments
    x: Input tensor.

    # Returns
    Binarized tensor.

    # References
    - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
      Representational Capability and Advanced
      Training Algorithm](http://arxiv.org/abs/1808.00278)
    """

    def grad(dy):
        return (1 - tf.abs(x)) * 2 * dy

    return sign(x), grad


def serialize(initializer):
    return tf.keras.utils.serialize_keras_object(initializer)


def deserialize(name, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="quantization function",
    )


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, str):
        return deserialize(str(identifier))
    if callable(identifier):
        return identifier
    raise ValueError(
        "Could not interpret quantization function identifier:", identifier
    )
