import pytest
import tensorflow as tf
import larq as lq


def test_summary(snapshot, capsys):
    model = tf.keras.models.Sequential(
        [
            lq.layers.QuantConv2D(
                32, (3, 3), kernel_quantizer="ste_sign", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            lq.layers.QuantConv2D(32, (3, 3), kernel_quantizer=lq.quantizers.SteTern()),
            tf.keras.layers.Dense(10),
        ]
    )
    lq.models.summary(model)
    captured = capsys.readouterr()
    snapshot.assert_match(captured.out)


def test_summary_invalid_model():
    with pytest.raises(ValueError):
        lq.models.summary(tf.keras.Model())
