import tensorflow as tf
import numpy as np

from larq.optimizers import Bop


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
        self.batch_previous_weights = {}
        self.epoch_previous_weights = {}
        self.update_freq = update_freq if update_freq != "batch" else 1

    def _maybe_log_and_store(self, storage, logs, should_log=True, should_store=True):
        if should_log or should_store:
            ops = []
            op_names = []
            for layer in self.model.layers:
                if hasattr(layer, "quantized_weights"):
                    for i, weight in enumerate(layer.quantized_weights):
                        ops.append(weight)
                        op_names.append(layer.name if i == 0 else f"{layer.name}_{i}")

            for key, value in zip(op_names, tf.keras.backend.batch_get_value(ops)):
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


class BopLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler for Boq optimizer.

  Arguments:
      schedule: a function that takes an epoch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.
  """

    def __init__(self, schedule, verbose=0):
        super(BopLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not isinstance(self.model.optimizer, Bop):
            raise ValueError(
                f"Expected larq.optimizers.Bop, received {type(self.model.optimizer)}."
            )

        if not hasattr(self.model.optimizer.fp_optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(tf.keras.backend.get_value(self.model.optimizer.fp_optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function ' "should be float."
            )
        tf.keras.backend.set_value(self.model.optimizer.fp_optimizer.lr, lr)
        if self.verbose > 0:
            print(
                "\nEpoch %05d: BoqLearningRateScheduler changing learning "
                "rate to %s." % (epoch + 1, lr)
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.fp_optimizer.lr)
