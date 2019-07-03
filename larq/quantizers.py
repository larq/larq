"""A Quantizer defines the way of transforming a full precision input to a
quantized output and the pseudo-gradient method used for the backwards pass."""

import tensorflow as tf
from larq import utils, math


@tf.custom_gradient
def _binarize_with_identity_grad(x):
    def grad(dy):
        return dy

    return math.sign(x), grad


@tf.custom_gradient
def _binarize_with_weighted_grad(x):
    def grad(dy):
        return (1 - tf.abs(x)) * 2 * dy

    return math.sign(x), grad


@utils.register_keras_custom_object
@utils.set_precision(1)
def ste_sign(x):
    r"""
    Sign binarization function.
    \\[
    q(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
    \end{cases}
    \\]

    The gradient is estimated using the Straight-Through Estimator
    (essentially the binarization is replaced by a clipped identity on the
    backward pass).
    \\[\frac{\partial q(x)}{\partial x} = \begin{cases}
      1 & \left|x\right| \leq 1 \\\
      0 & \left|x\right| > 1
    \end{cases}\\]

    ```plot-activation
    quantizers.ste_sign
    ```

    # Arguments
    x: Input tensor.

    # Returns
    Binarized tensor.

    # References
    - [Binarized Neural Networks: Training Deep Neural Networks with Weights and
      Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)
    """

    x = tf.clip_by_value(x, -1, 1)

    return _binarize_with_identity_grad(x)


@utils.register_keras_custom_object
@utils.set_precision(1)
def magnitude_aware_sign(x):
    r"""
    Magnitude-aware sign for Bi-Real Net.

    ```plot-activation
    quantizers.magnitude_aware_sign
    ```

    # Arguments
    x: Input tensor

    # Returns
    Scaled binarized tensor (with values in $\{-a, a\}$, where $a$ is a float).

    # References
    - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
      Representational Capability and Advanced Training
      Algorithm](https://arxiv.org/abs/1808.00278)

    """
    scale_factor = tf.reduce_mean(tf.abs(x), axis=list(range(len(x.shape) - 1)))

    return tf.stop_gradient(scale_factor) * ste_sign(x)


@utils.register_keras_custom_object
@utils.set_precision(1)
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
    \\[\frac{\partial q(x)}{\partial x} = \begin{cases}
      (2 - 2 \left|x\right|) & \left|x\right| \leq 1 \\\
      0 & \left|x\right| > 1
    \end{cases}
    \\]

    ```plot-activation
    quantizers.approx_sign
    ```

    # Arguments
    x: Input tensor.

    # Returns
    Binarized tensor.

    # References
    - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
      Representational Capability and Advanced
      Training Algorithm](http://arxiv.org/abs/1808.00278)
    """

    x = tf.clip_by_value(x, -1, 1)

    return _binarize_with_weighted_grad(x)


@utils.register_keras_custom_object
@utils.set_precision(2)
class SteTern:
    r"""
    Ternarization function.
    \\[
    q(x) = \begin{cases}
    +1 & x > \Delta \\\
    0 & |x| > \Delta \\\
    -1 & x < \Delta
    \end{cases}
    \\]

    where $\Delta$ is defined as the threshold and can be passed as an argument,
    or can be calculated as per the Ternary Weight Networks original paper, such that

    \\[
    \Delta = \frac{0.7}{n} \sum_{i=1}^{n} |W_i|
    \\]
    where we assume that $W_i$ is generated from a normal distribution.

    The gradient is estimated using the Straight-Through Estimator
    (essentially the Ternarization is replaced by a clipped identity on the
    backward pass).
    \\[\frac{\partial q(x)}{\partial x} = \begin{cases}
    1 & \left|x\right| \leq 1 \\\
    0 & \left|x\right| > 1
    \end{cases}\\]

    ```plot-activation
    quantizers.SteTern
    ```

    # Arguments
    x: Input tensor.
    threshold_value: The value for the threshold, $\Delta$.
    ternary_weight_networks: Boolean of whether to use the Ternary Weight Networks threshold calculation.

    # Returns
    Ternarized tensor.

    # References
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    """

    def __init__(self, threshold_value=0.1, ternary_weight_networks=False):
        self.threshold_value = threshold_value
        self.ternary_weight_networks = ternary_weight_networks

    def __call__(self, x):
        x = tf.clip_by_value(x, -1, 1)
        if self.ternary_weight_networks:
            threshold = self.threshold_twn(x)
        else:
            threshold = self.threshold_value

        @tf.custom_gradient
        def _ternarize_with_identity_grad(x):
            def grad(dy):
                return dy

            return (tf.sign(tf.sign(x + threshold) + tf.sign(x - threshold)), grad)

        return _ternarize_with_identity_grad(x)

    def threshold_twn(self, x):
        return 0.7 * tf.reduce_sum(tf.abs(x)) / tf.cast(tf.size(x), x.dtype)

    def get_config(self):
        return {
            "threshold_value": self.threshold_value,
            "ternary_weight_networks": self.ternary_weight_networks,
        }


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
        f"Could not interpret quantization function identifier: {identifier}"
    )
