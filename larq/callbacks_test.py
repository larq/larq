import numpy as np
import tensorflow as tf
import larq as lq
from tensorflow.python.keras import keras_parameterized


class LogHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batches = []
        self.epochs = []

    def _store_logs(self, storage, batch_or_epoch, logs={}):
        if [key for key in logs if "changed_quantization_ration" in key]:
            storage.append(batch_or_epoch)

    def on_batch_end(self, batch, logs={}):
        self._store_logs(self.batches, batch, logs)

    def on_epoch_end(self, epoch, logs={}):
        self._store_logs(self.epochs, epoch, logs)


@keras_parameterized.run_all_keras_modes
class LayersTest(keras_parameterized.TestCase):
    def test_quantization_logger(self):
        model = tf.keras.models.Sequential(
            [
                lq.layers.QuantSeparableConv1D(
                    1,
                    1,
                    depthwise_quantizer="ste_sign",
                    pointwise_quantizer="ste_sign",
                    input_shape=(3, 3),
                ),
                tf.keras.layers.Flatten(),
                lq.layers.QuantDense(2, kernel_quantizer="ste_sign"),
            ]
        )
        model.compile(loss="mse", optimizer="sgd")

        logger = lq.callbacks.QuantizationLogger(update_freq=5)
        history = LogHistory()

        x = np.ones((20, 3, 3))
        y = np.zeros((20, 2))
        model.fit(x, y, batch_size=1, epochs=3, callbacks=[logger, history])
        assert history.batches == [4, 9, 14, 19] * 3
        assert history.epochs == [0, 1, 2]
