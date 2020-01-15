from typing import Callable, Dict, Optional

import tensorflow as tf
from tensorflow import keras


class HyperparameterScheduler(tf.keras.callbacks.Callback):
    """Generic hyperparameter scheduler.

    !!! example
        ```python
        bop = lq.optimizers.Bop(threshold=1e-6, gamma=1e-3)
        adam = tf.keras.optimizers.Adam(0.01)
        optimizer = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, bop), default_optimizer=adam,
        )
        callbacks = [
            HyperparameterScheduler(lambda x: 0.001 * (0.1 ** (x // 30)), "gamma", bop)
        ]
        ```
    # Arguments
    optimizer: the optimizer that contains the hyperparameter that will be scheduled.
        Defaults to `self.model.optimizer` if `optimizer == None`.
    schedule: a function that takes an epoch index as input
        (integer, indexed from 0) and returns a new hyperparameter as output.
    hyperparameter: str. the name of the hyperparameter to be scheduled.
    unit: str (optional), what interval unit to change the hyperparameter at. Can be
        either "epoch" (default) or "step".
    verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(
        self,
        schedule: Callable,
        hyperparameter: str,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        unit: Optional[str] = "epoch",
        verbose: Optional[int] = 0,
    ):
        super(HyperparameterScheduler, self).__init__()
        self.optimizer = optimizer
        self.schedule = schedule
        self.hyperparameter = hyperparameter
        self.verbose = verbose

        # TODO: may want to make this a boolean instead for efficient comparison,
        # need to test impact on speed.
        if unit == "epoch" or unit == "step":
            self.unit = unit
        else:
            raise ValueError(
                f"HyperparameterScheduler.unit can only be 'step' or 'epoch'. Received value '{unit}'"
            )

    def set_model(self, model: keras.models.Model):
        super().set_model(model)
        if self.optimizer is None:
            # It is not possible for a model to reach this state and not have
            # an optimizer, so we can safely access it here.
            self.optimizer = model.optimizer

        if not hasattr(self.optimizer, self.hyperparameter):
            raise ValueError(
                f'Optimizer must have a "{self.hyperparameter}" attribute.'
            )

    def set_hyperparameter(self, t: int):
        hp = getattr(self.optimizer, self.hyperparameter)
        try:  # new API
            hyperparameter_val = tf.keras.backend.get_value(hp)
            hyperparameter_val = self.schedule(t, hyperparameter_val)
        except TypeError:  # Support for old API for backward compatibility
            hyperparameter_val = self.schedule(t)

        tf.keras.backend.set_value(hp, hyperparameter_val)
        return hp

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        if self.unit == "epoch":
            hp = self.set_hyperparameter(epoch)

            if self.verbose > 0:
                print(
                    f"Epoch {epoch + 1}: {self.hyperparameter} changing "
                    + f"to {tf.keras.backend.get_value(hp)}."
                )

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        if self.unit == "step":
            # We use optimizer.iterations (i.e. global step), since batch only
            # reflects the batch index in the current epoch.
            self.set_hyperparameter(self.optimizer.iterations)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        hp = getattr(self.optimizer, self.hyperparameter)
        logs[self.hyperparameter] = tf.keras.backend.get_value(hp)
