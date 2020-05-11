"""Functions from the `constraints` module allow setting constraints
(eg. weight clipping) on network parameters during optimization.

The penalties are applied on a per-layer basis. The exact API will depend on the layer,
but the layers `QuantDense`, `QuantConv1D`, `QuantConv2D` and `QuantConv3D` have a
unified API.

These layers expose 2 keyword arguments:

- `kernel_constraint` for the main weights matrix
- `bias_constraint` for the bias.

```python
import larq as lq

lq.layers.QuantDense(64, kernel_constraint="weight_clip")
lq.layers.QuantDense(64, kernel_constraint=lq.constraints.WeightClip(2.))
```
"""

from typing import Any, Mapping

import tensorflow as tf

from larq import utils


@utils.register_keras_custom_object
class WeightClip(tf.keras.constraints.Constraint):
    """Weight Clip constraint

    Constrains the weights incident to each hidden unit
    to be between `[-clip_value, clip_value]`.

    # Arguments
        clip_value: The value to clip incoming weights.
    """

    def __init__(self, clip_value: float = 1):
        self.clip_value = clip_value

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(x, -self.clip_value, self.clip_value)

    def get_config(self) -> Mapping[str, Any]:
        return {"clip_value": self.clip_value}


# Aliases
@utils.register_keras_custom_object
class weight_clip(WeightClip):
    pass
