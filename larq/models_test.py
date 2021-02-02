import numpy as np
import pytest
import tensorflow as tf
from packaging import version

import larq as lq
from larq.models import ModelProfile


class ToyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = lq.layers.QuantConv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_quantizer="ste_sign",
            input_shape=(64, 64, 1),
            padding="same",
        )
        self.pool = tf.keras.layers.GlobalAvgPool2D()
        self.dense = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        return self.dense(self.pool(self.conv(inputs)))


def get_functional_model():
    input = tf.keras.Input((32, 32, 3))
    x = lq.layers.QuantConv2D(
        filters=32,
        kernel_size=(3, 3),
        kernel_quantizer="ste_sign",
        padding="same",
    )(input)
    y, z = tf.split(x, 2, axis=-1)
    x = tf.concat([y, z], axis=-1)
    return tf.keras.Model(input, x, name="toy_model")


def get_profile_model():
    return tf.keras.models.Sequential(
        [
            lq.layers.QuantConv2D(
                filters=32,
                kernel_size=(3, 3),
                kernel_quantizer="ste_sign",
                input_shape=(64, 64, 1),
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            lq.layers.QuantDepthwiseConv2D(
                kernel_size=3,
                strides=(3, 3),
                input_quantizer=lq.quantizers.SteTern(),
                depthwise_quantizer=lq.quantizers.SteTern(),
                padding="same",
                pad_values=1.0,
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantSeparableConv2D(
                32,
                (3, 3),
                input_quantizer="ste_sign",
                depthwise_quantizer="ste_sign",
                pointwise_quantizer="ste_sign",
                padding="same",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, trainable=False),
        ]
    )


def test_model_profile():
    profile = ModelProfile(get_profile_model())
    assert len(profile.layer_profiles) == 7


def test_layer_profile():
    profile = ModelProfile(get_profile_model())

    kernel_count = [
        32 * 3 * 3 * 1,
        0,
        32 * 3 * 3,
        0,
        32 * 3 * 3 * 1 + 32 * 1 * 1 * 32,
        0,
        32 * 11 * 11 * 10,
    ]
    bias_count = [32, 0, 0, 64, 32, 0, 10]
    param_count = [k + b for k, b in zip(kernel_count, bias_count)]
    memory = [  # bits * (c * w * h * b) + bits * bias
        1 * (32 * 3 * 3 * 1) + 32 * 32,
        0,
        2 * (32 * 3 * 3),
        32 * (2 * 32),
        1 * (32 * 3 * 3 * 1 + 32 * 1 * 1 * 32) + 32 * 32,
        0,
        32 * (32 * 11 * 11 * 10 + 10),
    ]
    int8_fp_weights_mem = [
        1 * (32 * 3 * 3 * 1) + 8 * 32,
        0,
        2 * (32 * 3 * 3),
        8 * (32 * 2),
        1 * (32 * 3 * 3 * 1 + 32 * 1 * 1 * 32) + 8 * 32,
        0,
        8 * (32 * 11 * 11 * 10 + 10),
    ]
    fp_equiv_mem = [32 * n for n in param_count]
    input_precision = [None, None, 2, None, 1, None, None]
    output_shape = [
        (-1, 64, 64, 32),
        (-1, 32, 32, 32),
        (-1, 11, 11, 32),
        (-1, 11, 11, 32),
        (-1, 11, 11, 32),
        (-1, 11 * 11 * 32),
        (-1, 10),
    ]
    output_pixels = [int(np.prod(os[1:-1])) for os in output_shape]
    unique_param_bidtwidths = [[1, 32], [], [2], [32], [1, 32], [], [32]]
    unique_op_precisions = [[32], [], [2], [], [1], [], [32]]
    mac_count = [params * pixels for params, pixels in zip(kernel_count, output_pixels)]
    bin_mac_count = [
        mc if (1 in pb and ip == 1) else 0
        for mc, pb, ip in zip(mac_count, unique_param_bidtwidths, input_precision)
    ]

    profiles = profile.layer_profiles
    for i in range(len(profiles)):
        print(f"Testing layer {i}...")
        assert profiles[i].input_precision == input_precision[i]
        assert profiles[i].output_shape == output_shape[i]
        assert profiles[i].output_pixels == output_pixels[i]
        assert profiles[i].weight_count() == param_count[i]
        assert profiles[i].unique_param_bidtwidths == unique_param_bidtwidths[i]
        assert profiles[i].unique_op_precisions == unique_op_precisions[i]
        assert profiles[i].memory == memory[i]
        assert profiles[i].fp_equivalent_memory == fp_equiv_mem[i]
        assert profiles[i].int8_fp_weights_memory == int8_fp_weights_mem[i]
        assert profiles[i].op_count("mac") == mac_count[i]
        assert profiles[i].op_count("mac", 1) == bin_mac_count[i]


def test_layer_profile_1d():
    model = tf.keras.models.Sequential(
        [
            lq.layers.QuantConv1D(
                filters=32,
                kernel_size=3,
                input_shape=(64, 6),
                kernel_quantizer="ste_sign",
                padding="same",
            ),
            tf.keras.layers.MaxPooling1D(2),
            lq.layers.QuantSeparableConv1D(
                filters=16,
                kernel_size=3,
                input_quantizer="ste_sign",
                depthwise_quantizer="ste_sign",
                pointwise_quantizer="ste_sign",
                padding="same",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, trainable=False),
        ]
    )
    profile = ModelProfile(model)

    kernel_count = [(32 * 3 * 6), 0, (32 * 3 + 16 * 32), 0, (16 * 32 * 10)]
    bias_count = [32, 0, 16, 0, 10]
    param_count = [k + b for k, b in zip(kernel_count, bias_count)]
    memory = [  # bits * (c * w * d) + bits * bias
        1 * (32 * 3 * 6) + 32 * 32,
        0,
        1 * (32 * 3 + 16 * 32) + 32 * 16,
        0,
        32 * (32 * 16 * 10 + 10),
    ]
    int8_fp_weights_mem = [
        1 * (32 * 3 * 6) + 8 * 32,
        0,
        1 * (32 * 3 + 16 * 32) + 8 * 16,
        0,
        8 * (32 * 16 * 10 + 10),
    ]
    fp_equiv_mem = [32 * n for n in param_count]
    input_precision = [None, None, 1, None, None]
    output_shape = [
        (-1, 64, 32),
        (-1, 32, 32),
        (-1, 32, 16),
        (-1, 32 * 16),
        (-1, 10),
    ]
    output_pixels = [int(np.prod(os[1:-1])) for os in output_shape]
    unique_param_bidtwidths = [[1, 32], [], [1, 32], [], [32]]
    unique_op_precisions = [[32], [], [1], [], [32]]
    mac_count = [params * pixels for params, pixels in zip(kernel_count, output_pixels)]
    bin_mac_count = [
        mc if (1 in pb and ip == 1) else 0
        for mc, pb, ip in zip(mac_count, unique_param_bidtwidths, input_precision)
    ]

    profiles = profile.layer_profiles
    for i in range(len(profiles)):
        print(f"Testing layer {i}...")
        assert profiles[i].input_precision == input_precision[i]
        assert profiles[i].output_shape == output_shape[i]
        assert profiles[i].output_pixels == output_pixels[i]
        assert profiles[i].weight_count() == param_count[i]
        assert profiles[i].unique_param_bidtwidths == unique_param_bidtwidths[i]
        assert profiles[i].unique_op_precisions == unique_op_precisions[i]
        assert profiles[i].memory == memory[i]
        assert profiles[i].fp_equivalent_memory == fp_equiv_mem[i]
        assert profiles[i].int8_fp_weights_memory == int8_fp_weights_mem[i]
        assert profiles[i].op_count("mac") == mac_count[i]
        assert profiles[i].op_count("mac", 1) == bin_mac_count[i]


def test_summary(snapshot, capsys):
    model = get_profile_model()
    lq.models.summary(model)
    captured = capsys.readouterr()
    snapshot.assert_match(captured.out)

    # A model with no weights
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Lambda(lambda x: tf.zeros(2), input_shape=(32, 32))]
    )
    lq.models.summary(model)
    captured = capsys.readouterr()
    snapshot.assert_match(captured.out)


def test_subclass_model_summary(snapshot, capsys):
    model = ToyModel()
    model.build((None, 32, 32, 3))
    lq.models.summary(model)
    captured = capsys.readouterr()
    snapshot.assert_match(captured.out)


def test_functional_model_summary(snapshot, capsys):
    lq.models.summary(get_functional_model())
    captured = capsys.readouterr()
    key = "2.4+" if version.parse(tf.__version__) >= version.parse("2.3.9") else "<2.4"
    snapshot.assert_match(captured.out.lower(), key)


def test_summary_invalid_model():
    with pytest.raises(ValueError):
        lq.models.summary(tf.keras.Model())


def test_bitsize_invalid_key():
    with pytest.raises(NotImplementedError):
        lq.models._bitsize_as_str(-1)


def test_number_as_readable_str_large():
    assert lq.models._number_as_readable_str(1e16) == "1.00E+16"


@pytest.fixture(autouse=True)
def run_around_tests():
    tf.keras.backend.clear_session()
    yield
