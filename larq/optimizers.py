import tensorflow as tf
import numpy as np
from larq import utils


@utils.register_keras_custom_object
class XavierLearningRateScaling(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for Xavier Learning Rate Scaling

    Scale the weights learning rates respectively with the weights initialization

    !!! note ""
        This is a wrapper and does not implement any optimization algorithm.

    !!! example
        ```python
        optimizer = lq.optimizers.XavierLearningRateScaling(
            tf.keras.optimizers.Adam(0.01), model
        )
        ```

    # Arguments
    optimizer: A `tf.keras.optimizers.Optimizer`
    model: A `tf.keras.Model`

    # References
    - [BinaryConnect: Training Deep Neural Networks with binary weights during
      propagations](https://arxiv.org/abs/1511.00363)
    """

    def __init__(self, optimizer, model):
        if int(tf.__version__[0]) == 2:
            raise NotImplementedError(
                "XavierLearningRateScaling is not supported by Tensorflow 2.0."
            )

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise ValueError(
                f"Expected tf.keras.optimizers.Optimizer, received {type(optimizer)}."
            )
        self.optimizer = optimizer

        if isinstance(model, tf.keras.Model):
            self.multipliers = {}
            for layer in model.layers:
                if hasattr(layer, "quantized_latent_weights"):
                    for weight in layer.quantized_latent_weights:
                        self.multipliers[weight.name] = self.get_lr_multiplier(weight)
        elif isinstance(model, dict):
            self.multipliers = model
        else:
            raise ValueError(f"Expected tf.keras.Model or dict, received {type(model)}")

    def get_lr_multiplier(self, weight):
        shape = weight.get_shape().as_list()
        n_input = shape[-2]
        n_output = shape[-1]
        if len(shape) == 4:
            kernelsize = np.prod(shape[:-2])
            coeff = 1.0 / np.sqrt(1.5 / ((kernelsize * (n_input + n_output))))
        elif len(shape) == 2:
            coeff = 1.0 / np.sqrt(1.5 / ((1.0 * (n_input + n_output))))
        else:
            raise NotImplementedError(
                "Xavier Learning rate scaling not implimented for this kernelsize"
            )
        return coeff

    def get_updates(self, loss, params):
        mult_lr_params = [p for p in params if p.name in self.multipliers]
        base_lr_params = [p for p in params if p.name not in self.multipliers]

        updates = []
        base_lr = self.optimizer.lr
        for param in mult_lr_params:
            self.optimizer.lr = base_lr * self.multipliers[param.name]
            updates.extend(self.optimizer.get_updates(loss, [param]))

        self.optimizer.lr = base_lr
        updates.extend(self.optimizer.get_updates(loss, base_lr_params))

        return updates

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def get_config(self):
        return {
            "optimizer": {
                "class_name": self.optimizer.__class__.__name__,
                "config": self.optimizer.get_config(),
            },
            "multipliers": self.multipliers,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, config["multipliers"])
