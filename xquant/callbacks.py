import tensorflow as tf
import numpy as np


class QuantizationLogger(tf.keras.callbacks.Callback):
    """Callback that adds quantization specific metrics.

    !!! note ""
        In order for metrics to be picked up by TensorBoard this callback needs to be
        applied before the TensorBoard callback and use the same update frequency.

    !!! example
        ```python
        callbacks = [
            QuantizationLogger(update_freq=100),
            tf.keras.callbacks.TensorBoard(update_freq=100),
        ]
        model.fit(X_train, Y_train, callbacks=callbacks)
        ```

    # Metrics
    - `changed_quantization_ration`: The ration of quantized weights in each layer that
        changed during the weight update.

    # Arguments
    update_freq: `'batch'` or integer. When using `'batch'`, computes the metrics after
        each batch. If using an integer the callback will compute the metrics every
        `update_freq` batches. Note that computing too frequently can slow down training.
    """

    def __init__(self, update_freq="batch"):
        self.previous_weights = {}
        self.update_freq = update_freq if update_freq != "batch" else 1

    def on_batch_end(self, batch, logs=None):
        ops = []
        op_names = []
        for layer in self.model.layers:
            if hasattr(layer, "quantized_weights"):
                for weight in layer.quantized_weights:
                    ops.append(weight)
                    op_names.append(weight.name)

        for key, value in zip(op_names, tf.keras.backend.batch_get_value(ops)):
            if batch > 0 and batch % self.update_freq == 0:
                logs[
                    f"changed_quantization_ration/{key.replace(':', '_')}"
                ] = 1 - np.mean(value == self.previous_weights[key])
            if (batch + 1) % self.update_freq == 0:
                self.previous_weights[key] = value
