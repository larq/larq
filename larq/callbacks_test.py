import numpy as np
import tensorflow as tf
import larq as lq
from tensorflow.python.keras import keras_parameterized


class LogHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.logs = []

    def on_batch_end(self, batch, logs={}):
        if [key for key in logs if "changed_quantization_ration" in key]:
            self.logs.append(batch)


@keras_parameterized.run_all_keras_modes
class LayersTest(keras_parameterized.TestCase):
    def test_quantization_logger(self):
        if int(tf.__version__[0]) == 2:
            # tf.keras.backend.batch_get_value() currently fails in TF 2 functions
            return
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

        x = np.ones((25, 3, 3))
        y = np.zeros((25, 2))
        model.fit(x, y, batch_size=1, epochs=1, callbacks=[logger, history])
        assert history.logs == [4, 9, 14, 19, 24]
