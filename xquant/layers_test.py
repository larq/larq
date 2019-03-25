import tensorflow as tf
import numpy as np
from absl.testing import parameterized
import xquant as xq

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils


def random_input(shape):
    for i, dim in enumerate(shape):
        if dim is None:
            shape[i] = np.random.randint(1, 4)
    data = 10 * np.random.random(shape) - 0.5
    return data.astype("float32")


parameterized_all_layers = parameterized.named_parameters(
    ("QuantDense", xq.layers.QuantDense, tf.keras.layers.Dense, (3, 2), dict(units=3)),
    (
        "QuantConv1D",
        xq.layers.QuantConv1D,
        tf.keras.layers.Conv1D,
        (2, 3, 7),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv2D",
        xq.layers.QuantConv2D,
        tf.keras.layers.Conv2D,
        (2, 3, 7, 6),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv3D",
        xq.layers.QuantConv3D,
        tf.keras.layers.Conv3D,
        (2, 3, 7, 6, 5),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv2DTranspose",
        xq.layers.QuantConv2DTranspose,
        tf.keras.layers.Conv2DTranspose,
        (2, 3, 7, 6),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv3DTranspose",
        xq.layers.QuantConv3DTranspose,
        tf.keras.layers.Conv3DTranspose,
        (2, 3, 7, 6, 5),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantLocallyConnected1D",
        xq.layers.QuantLocallyConnected1D,
        tf.keras.layers.LocallyConnected1D,
        (2, 8, 5),
        dict(filters=4, kernel_size=3),
    ),
    (
        "QuantLocallyConnected2D",
        xq.layers.QuantLocallyConnected2D,
        tf.keras.layers.LocallyConnected2D,
        (8, 6, 10, 4),
        dict(filters=3, kernel_size=3),
    ),
)


@keras_parameterized.run_all_keras_modes
class LayersTest(keras_parameterized.TestCase):
    @parameterized_all_layers
    def test_binarization(self, quantized_layer, layer, input_shape, kwargs):
        input_data = random_input(input_shape)
        random_weight = np.random.random() - 0.5

        quant_output = testing_utils.layer_test(
            quantized_layer,
            kwargs=dict(
                **kwargs,
                kernel_quantizer="ste_sign",
                input_quantizer="ste_sign",
                kernel_initializer=tf.keras.initializers.constant(random_weight),
            ),
            input_data=input_data,
        )

        fp_output = testing_utils.layer_test(
            layer,
            kwargs=dict(
                **kwargs,
                kernel_initializer=tf.keras.initializers.constant(
                    np.sign(random_weight)
                ),
            ),
            input_data=np.sign(input_data),
        )

        self.assertAllClose(quant_output, fp_output)

    @parameterized.named_parameters(
        (
            "QuantSeparableConv1D",
            xq.layers.QuantSeparableConv1D,
            tf.keras.layers.SeparableConv1D,
            (2, 3, 7),
        ),
        (
            "QuantSeparableConv2D",
            xq.layers.QuantSeparableConv2D,
            tf.keras.layers.SeparableConv2D,
            (2, 3, 7, 6),
        ),
    )
    def test_separable_layers(self, quantized_layer, layer, input_shape):
        input_data = random_input(input_shape)
        random_d_kernel = np.random.random() - 0.5
        random_p_kernel = np.random.random() - 0.5

        quant_output = testing_utils.layer_test(
            quantized_layer,
            kwargs=dict(
                filters=3,
                kernel_size=3,
                depthwise_quantizer="ste_sign",
                pointwise_quantizer="ste_sign",
                input_quantizer="ste_sign",
                depthwise_initializer=tf.keras.initializers.constant(random_d_kernel),
                pointwise_initializer=tf.keras.initializers.constant(random_p_kernel),
            ),
            input_data=input_data,
        )

        fp_output = testing_utils.layer_test(
            layer,
            kwargs=dict(
                filters=3,
                kernel_size=3,
                depthwise_initializer=tf.keras.initializers.constant(
                    np.sign(random_d_kernel)
                ),
                pointwise_initializer=tf.keras.initializers.constant(
                    np.sign(random_p_kernel)
                ),
            ),
            input_data=np.sign(input_data),
        )

        self.assertAllClose(quant_output, fp_output)
