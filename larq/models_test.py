import pytest
import tensorflow as tf
import larq as lq


def test_summary(snapshot, capsys):
    model = tf.keras.models.Sequential(
        [
            lq.layers.QuantConv2D(
                32,
                (3, 3),
                kernel_quantizer="ste_sign",
                input_shape=(64, 64, 1),
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            lq.layers.QuantDepthwiseConv2D(
                32,
                (3, 3),
                input_quantizer=lq.quantizers.SteTern(),
                depthwise_quantizer=lq.quantizers.SteTern(),
                padding="same",
            ),
            lq.layers.QuantSeparableConv2D(
                32,
                (3, 3),
                input_quantizer="ste_sign",
                depthwise_quantizer="ste_sign",
                pointwise_quantizer="ste_sign",
                padding="same",
            ),
            tf.keras.layers.Dense(10),
        ]
    )
    lq.models.summary(model)
    captured = capsys.readouterr()
    snapshot.assert_match(captured.out)


def test_summary_invalid_model():
    with pytest.raises(ValueError):
        lq.models.summary(tf.keras.Model())
