"""Activations can either be used through an `Activation` layer, or through the
`activation` argument supported by all forward layers:

```python
import tensorflow as tf
import xquant as xq

model.add(xq.layers.QuantDense(64))
model.add(tf.keras.layers.Activation('hard_tanh'))
```

This is equivalent to:

```python
model.add(xq.layers.QuantDense(64, activation='hard_tanh'))
```

You can also pass an element-wise TensorFlow function as an activation:

```python
model.add(xq.layers.QuantDense(64, activation=xq.activations.hard_tanh))
```
"""

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
