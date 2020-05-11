"""Activations can either be used through an `Activation` layer, or through the
`activation` argument supported by all forward layers:

```python
import tensorflow as tf
import larq as lq

model.add(lq.layers.QuantDense(64))
model.add(tf.keras.layers.Activation('hard_tanh'))
```

This is equivalent to:

```python
model.add(lq.layers.QuantDense(64, activation='hard_tanh'))
```

You can also pass an element-wise TensorFlow function as an activation:

```python
model.add(lq.layers.QuantDense(64, activation=lq.activations.hard_tanh))
```
"""

import tensorflow as tf

from larq import utils


@utils.register_keras_custom_object
def hard_tanh(x: tf.Tensor) -> tf.Tensor:
    """Hard tanh activation function.
    ```plot-activation
    activations.hard_tanh
    ```

    # Arguments
        x: Input tensor.

    # Returns
        Hard tanh activation.
    """
    return tf.clip_by_value(x, -1, 1)


@utils.register_keras_custom_object
def leaky_tanh(x: tf.Tensor, alpha: float = 0.2) -> tf.Tensor:
    r"""Leaky tanh activation function.
    Similar to hard tanh, but with non-zero slopes as in leaky ReLU.

    ```plot-activation
    activations.leaky_tanh
    ```

    # Arguments
        x: Input tensor.
        alpha: Slope of the activation function outside of [-1, 1].

    # Returns
        Leaky tanh activation.
    """
    return (
        tf.clip_by_value(x, -1, 1)
        + (tf.math.maximum(x, 1) - 1) * alpha
        + (tf.math.minimum(x, -1) + 1) * alpha
    )
