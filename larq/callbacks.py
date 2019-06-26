import tensorflow as tf
import numpy as np


class QuantizationLogger(tf.keras.callbacks.Callback):
    """Callback that adds quantization specific metrics.

    !!! note ""
        In order for metrics to be picked up by TensorBoard this callback needs to be
        applied before the TensorBoard callback and use the same update frequency.

    !!! example
        ```python
        callbacks = [QuantizationLogger(), tf.keras.callbacks.TensorBoard()]
        model.fit(X_train, Y_train, callbacks=callbacks)
        ```

    # Metrics
    - `changed_quantization_ration`: The ration of quantized weights in each layer that
        changed during the weight update.

    # Arguments
    update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, computes the
        metrics after each batch. The same applies for `'epoch'`. If using an integer
        the callback will compute the metrics every `update_freq` batches.
        Note that computing too frequently can slow down training.
    """

    def __init__(self, update_freq="epoch"):
        super().__init__()
        self.batch_previous_weights = {}
        self.epoch_previous_weights = {}
        self.update_freq = update_freq if update_freq != "batch" else 1
        self._quantized_weights = []
        self._quantized_weight_names = []

    def set_model(self, model):
        super().set_model(model)
        for layer in model.layers:
            if hasattr(layer, "quantized_weights"):
                for i, weight in enumerate(layer.quantized_weights):
                    self._quantized_weights.append(weight)
                    self._quantized_weight_names.append(
                        layer.name if i == 0 else f"{layer.name}_{i}"
                    )

    def _maybe_log_and_store(self, storage, logs, should_log=True, should_store=True):
        if should_log or should_store:
            values = tf.keras.backend.batch_get_value(self._quantized_weights)
            for key, value in zip(self._quantized_weight_names, values):
                value = value.astype(np.int8)
                if should_log:
                    logs[f"changed_quantization_ration/{key.replace(':', '_')}"] = 1 - (
                        np.count_nonzero(value == storage[key]) / value.size
                    )
                if should_store:
                    storage[key] = value

            if should_log and not should_store:
                # We don't need it in the next batch anymore
                storage = {}

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != "epoch":
            self._maybe_log_and_store(
                self.batch_previous_weights,
                logs,
                should_log=batch > 0 and (batch + 1) % self.update_freq == 0,
                should_store=(batch + 2) % self.update_freq == 0,
            )

    def on_train_begin(self, logs=None):
        self._maybe_log_and_store(self.epoch_previous_weights, logs, should_log=False)

    def on_epoch_end(self, epoch, logs=None):
        self._maybe_log_and_store(self.epoch_previous_weights, logs)


class HyperparameterScheduler(tf.keras.callbacks.Callback):
    """Generic hyperparameter scheduler.

    # Arguments:
    schedule: a function that takes an epoch index as input
        (integer, indexed from 0) and returns a new hyperparameter as output.
    hyperparameter: str. the name of the hyperparameter to be scheduled.
    verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, hyperparameter, verbose=0):
        super(HyperparameterScheduler, self).__init__()
        self.schedule = schedule
        self.hyperparameter = hyperparameter
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, self.hyperparameter):
            raise ValueError(
                f'Optimizer must have a "{self.hyperparameter}" attribute.'
            )

        hp = getattr(self.model.optimizer, self.hyperparameter)
        try:  # new API
            hyperparameter_val = tf.keras.backend.get_value(hp)
            hyperparameter_val = self.schedule(epoch, hyperparameter_val)
        except TypeError:  # Support for old API for backward compatibility
            hyperparameter_val = self.schedule(epoch)

        tf.keras.backend.set_value(hp, hyperparameter_val)

        if self.verbose > 0:
            print(
                f"Epoch {epoch + 1}: {self.hyperparameter} changning to {tf.keras.backend.get_value(hp)}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        hp = getattr(self.model.optimizer, self.hyperparameter)
        logs[self.hyperparameter] = tf.keras.backend.get_value(hp)
