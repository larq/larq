import tensorflow as tf
import numpy as np


class XavierLearningRateScaling(tf.keras.optimizers.Optimizer):
    r"""
    Xavier Learning Rate Scaling

    Scale the weights learning rates respectively with the Weights initialisation

    # References
    - [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)
    """

    def __init__(self, model, optimizer, **kwargs):
        self._class = optimizer
        self._optimizer = optimizer(**kwargs)

        self.multipliers = {}
        for layer in model.layers:
            if hasattr(layer, "quantized_latent_weights"):
                for weight in layer.quantized_latent_weights:
                    self.multipliers[weight.name] = self.get_lr_multiplier(weight)

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
            raise NotImplementedError()
        return coeff

    def get_updates(self, loss, params):
        mult_lr_params = [p for p in params if p.name in self.multipliers]
        base_lr_params = [p for p in params if p.name not in self.multipliers]

        updates = []
        base_lr = self._optimizer.lr
        for param in mult_lr_params:
            self._optimizer.lr = base_lr * self.multipliers[param.name]
            updates.extend(self._optimizer.get_updates(loss, [param]))

        self._optimizer.lr = base_lr
        updates.extend(self._optimizer.get_updates(loss, base_lr_params))

        return updates
