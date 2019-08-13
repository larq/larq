import pytest
import tensorflow as tf
import larq as lq
from larq.models import ModelProfile
import numpy as np


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
                use_bias=False,
            ),
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
    assert len(profile.layer_profiles) == 6


def test_layer_profile():
    profile = ModelProfile(get_profile_model())

    kernel_count = [
        32 * 3 * 3 * 1,
        0,
        32 * 3 * 3,
        32 * 3 * 3 * 1 + 32 * 1 * 1 * 32,
        0,
        32 * 11 * 11 * 10,
    ]
    bias_count = [32, 0, 0, 32, 0, 10]
    param_count = [k + b for k, b in zip(kernel_count, bias_count)]
    memory = [
        1 * (32 * 3 * 3 * 1) + 32 * 32,
        0,
        2 * (32 * 3 * 3),
        1 * (32 * 3 * 3 * 1 + 32 * 1 * 1 * 32) + 32 * 32,
        0,
        32 * (32 * 11 * 11 * 10 + 10),
    ]
    fp_equiv_mem = [32 * n for n in param_count]
    input_precision = ["-", "-", 2, 1, "-", "-"]
    output_shape = [
        (-1, 64, 64, 32),
        (-1, 32, 32, 32),
        (-1, 11, 11, 32),
        (-1, 11, 11, 32),
        (-1, 11 * 11 * 32),
        (-1, 10),
    ]
    output_pixels = [int(np.prod(os[1:-1])) for os in output_shape]
    unique_param_bidtwidths = [[1, 32], [], [2], [1, 32], [], [32]]
    unique_op_precisions = [[32], [], [2], [1], [], [32]]
    mac_count = [params * pixels for params, pixels in zip(kernel_count, output_pixels)]
    bin_mac_count = [
        mc if (1 in pb and ip == 1) else 0
        for mc, pb, ip in zip(mac_count, unique_param_bidtwidths, input_precision)
    ]

    profiles = profile.layer_profiles
    for i in range(len(profiles)):
        print(f"Testing layer {i}...")
        assert profiles[i].input_precision() == input_precision[i]
        assert profiles[i].output_shape == output_shape[i]
        assert profiles[i].output_pixels == output_pixels[i]
        assert profiles[i].weight_count() == param_count[i]
        assert profiles[i].unique_param_bidtwidths == unique_param_bidtwidths[i]
        assert profiles[i].unique_op_precisions == unique_op_precisions[i]
        assert profiles[i].memory == memory[i]
        assert profiles[i].fp_equivalent_memory == fp_equiv_mem[i]
        assert profiles[i].op_count("mac") == mac_count[i]
        assert profiles[i].op_count("mac", 1) == bin_mac_count[i]


def test_summary(snapshot, capsys):
    model = get_profile_model()
    lq.models.summary(model)
    captured = capsys.readouterr()
    snapshot.assert_match(captured.out)


def test_summary_invalid_model():
    with pytest.raises(ValueError):
        lq.models.summary(tf.keras.Model())


@pytest.fixture(autouse=True)
def run_around_tests():
    tf.keras.backend.clear_session()
    yield
