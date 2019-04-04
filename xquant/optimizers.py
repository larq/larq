import tensorflow as tf


class XavierLearningRateScaling(tf.keras.optimizers.Optimizer):
    def __init__(self, model, optimizer, **kwargs):
        self._class = optimizer
        self._optimizer = optimizer(**kwargs)

        self.multipliers = {}
        for layer in model.layers:
            if hasattr(layer, "quantized_latent_weights"):
                for weight in layer.quantized_latent_weights:
                    self.multipliers[weight.name] = self.get_lr_multiplier(weight)

    def get_lr_multiplier(self, weight):
        return 1

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
