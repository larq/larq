import pytest
import tensorflow as tf
import larq as lq


def test_summary():
    model = tf.keras.models.Sequential(
        [
            lq.layers.QuantConv2D(
                32, (3, 3), kernel_quantizer="ste_sign", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
        ]
    )
    lq.models.summary(model)


def test_summary_invalid_model():
    with pytest.raises(ValueError):
        lq.models.summary(tf.keras.Model())
