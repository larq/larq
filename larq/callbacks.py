from typing import Any, Callable, MutableMapping, Optional

from tensorflow import keras


class HyperparameterScheduler(keras.callbacks.Callback):
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
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new hyperparameter as output.
        hyperparameter: str. the name of the hyperparameter to be scheduled.
        optimizer: the optimizer that contains the hyperparameter that will be scheduled.
            Defaults to `self.model.optimizer` if `optimizer == None`.
        update_freq: str (optional), denotes on what update_freq to change the
            hyperparameter. Can be either "epoch" (default) or "step".
        verbose: int. 0: quiet, 1: update messages.
        log_name: str (optional), under which name to log this hyperparameter to
            Tensorboard. If `None`, defaults to `hyperparameter`. Use this if you have
            several schedules for the same hyperparameter on different optimizers.
    """

    def __init__(
        self,
        schedule: Callable,
        hyperparameter: str,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        update_freq: str = "epoch",
        verbose: int = 0,
        log_name: Optional[str] = None,
    ):
        super(HyperparameterScheduler, self).__init__()
        self.optimizer = optimizer
        self.schedule = schedule
        self.hyperparameter = hyperparameter
        self.log_name = log_name or hyperparameter
        self.verbose = verbose

        if update_freq not in ["epoch", "step"]:
            raise ValueError(
                "HyperparameterScheduler.update_freq can only be 'step' or 'epoch'."
                f" Received value '{update_freq}'"
            )

        self.update_freq = update_freq

    def set_model(self, model: keras.models.Model) -> None:
        super().set_model(model)
        if self.optimizer is None:
            # It is not possible for a model to reach this state and not have
            # an optimizer, so we can safely access it here.
            self.optimizer = model.optimizer

        if not hasattr(self.optimizer, self.hyperparameter):
            raise ValueError(
                f'Optimizer must have a "{self.hyperparameter}" attribute.'
            )

    def set_hyperparameter(self, t: int) -> Any:
        hp = getattr(self.optimizer, self.hyperparameter)
        try:  # new API
            hyperparameter_val = keras.backend.get_value(hp)
            hyperparameter_val = self.schedule(t, hyperparameter_val)
        except TypeError:  # Support for old API for backward compatibility
            hyperparameter_val = self.schedule(t)
        keras.backend.set_value(hp, hyperparameter_val)
        return hp

    def on_batch_begin(
        self, batch: int, logs: Optional[MutableMapping[str, Any]] = None
    ) -> None:
        if not self.update_freq == "step":
            return

        # We use optimizer.iterations (i.e. global step), since batch only
        # reflects the batch index in the current epoch.
        batch = keras.backend.get_value(self.optimizer.iterations)
        hp = self.set_hyperparameter(batch)

        if self.verbose > 0:
            print(
                f"Batch {batch}: {self.log_name} is now {keras.backend.get_value(hp)}."
            )

    def on_epoch_begin(
        self, epoch: int, logs: Optional[MutableMapping[str, Any]] = None
    ) -> None:
        if not self.update_freq == "epoch":
            return

        hp = self.set_hyperparameter(epoch)

        if self.verbose > 0:
            print(
                f"Epoch {epoch}: {self.log_name} is now {keras.backend.get_value(hp)}."
            )

    def on_epoch_end(
        self, epoch: int, logs: Optional[MutableMapping[str, Any]] = None
    ) -> None:
        logs = logs or {}
        hp = getattr(self.optimizer, self.hyperparameter)
        logs[self.log_name] = keras.backend.get_value(hp)
