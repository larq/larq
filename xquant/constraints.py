import tensorflow as tf
from xquant import utils


@utils.register_keras_custom_object
class WeightClip(tf.keras.constraints.Constraint):
    """Weight Clip constraint

    Constrains the weights incident to each hidden unit
    to be between `[-clip_value, clip_value]`.

    # Arguments
    clip_value: The value to clip incoming weights.
    """

    def __init__(self, clip_value=1):
        self.clip_value = clip_value

    def __call__(self, x):
        return tf.clip_by_value(x, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}


# Aliases
@utils.register_keras_custom_object
class weight_clip(WeightClip):
    pass
