import tensorflow as tf
import numpy as np
from absl.testing import parameterized
import larq as lq
import pytest
import inspect

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils


def random_input(shape):
    for i, dim in enumerate(shape):
        if dim is None:
            shape[i] = np.random.randint(1, 4)
    data = 10 * np.random.random(shape) - 0.5
    return data.astype("float32")


parameterized_all_layers = parameterized.named_parameters(
    ("QuantDense", lq.layers.QuantDense, tf.keras.layers.Dense, (3, 2), dict(units=3)),
    (
        "QuantConv1D",
        lq.layers.QuantConv1D,
        tf.keras.layers.Conv1D,
        (2, 3, 7),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv2D",
        lq.layers.QuantConv2D,
        tf.keras.layers.Conv2D,
        (2, 3, 7, 6),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv3D",
        lq.layers.QuantConv3D,
        tf.keras.layers.Conv3D,
        (2, 3, 7, 6, 5),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv2DTranspose",
        lq.layers.QuantConv2DTranspose,
        tf.keras.layers.Conv2DTranspose,
        (2, 3, 7, 6),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantConv3DTranspose",
        lq.layers.QuantConv3DTranspose,
        tf.keras.layers.Conv3DTranspose,
        (2, 3, 7, 6, 5),
        dict(filters=2, kernel_size=3),
    ),
    (
        "QuantLocallyConnected1D",
        lq.layers.QuantLocallyConnected1D,
        tf.keras.layers.LocallyConnected1D,
        (2, 8, 5),
        dict(filters=4, kernel_size=3),
    ),
    (
        "QuantLocallyConnected2D",
        lq.layers.QuantLocallyConnected2D,
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
            lq.layers.QuantSeparableConv1D,
            tf.keras.layers.SeparableConv1D,
            (2, 3, 7),
        ),
        (
            "QuantSeparableConv2D",
            lq.layers.QuantSeparableConv2D,
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


def test_layer_warns(caplog):
    lq.layers.QuantDense(5, kernel_quantizer="ste_sign")
    assert len(caplog.records) == 1
    assert "kernel_constraint" in caplog.text


def test_layer_does_not_warn(caplog):
    lq.layers.QuantDense(
        5, kernel_quantizer="ste_sign", kernel_constraint="weight_clip"
    )
    assert caplog.records == []


def test_separable_layer_warns(caplog):
    lq.layers.QuantSeparableConv2D(
        3, 3, depthwise_quantizer="ste_sign", pointwise_quantizer="ste_sign"
    )
    assert len(caplog.records) == 2
    assert "depthwise_constraint" in caplog.text
    assert "pointwise_constraint" in caplog.text


def test_separable_layer_does_not_warn(caplog):
    lq.layers.QuantSeparableConv2D(
        3,
        3,
        depthwise_quantizer="ste_sign",
        pointwise_quantizer="ste_sign",
        depthwise_constraint="weight_clip",
        pointwise_constraint="weight_clip",
    )
    assert caplog.records == []


@pytest.mark.parametrize(
    "quant_layer,layer",
    [
        (lq.layers.QuantDense, tf.keras.layers.Dense),
        (lq.layers.QuantConv1D, tf.keras.layers.Conv1D),
        (lq.layers.QuantConv2D, tf.keras.layers.Conv2D),
        (lq.layers.QuantConv3D, tf.keras.layers.Conv3D),
        (lq.layers.QuantConv2DTranspose, tf.keras.layers.Conv2DTranspose),
        (lq.layers.QuantConv3DTranspose, tf.keras.layers.Conv3DTranspose),
        (lq.layers.QuantLocallyConnected1D, tf.keras.layers.LocallyConnected1D),
        (lq.layers.QuantLocallyConnected2D, tf.keras.layers.LocallyConnected2D),
    ],
)
def test_layer_kwargs(quant_layer, layer):
    quant_params = inspect.signature(quant_layer).parameters
    params = inspect.signature(layer).parameters

    quant_params_list = list(quant_params.keys())
    params_list = list(params.keys())

    for p in (
        "input_quantizer",
        "kernel_quantizer",
        "depthwise_quantizer",
        "pointwise_quantizer",
    ):
        try:
            quant_params_list.remove(p)
        except:
            pass
    assert quant_params_list == params_list

    for param in params_list:
        assert quant_params.get(param).default == params.get(param).default
