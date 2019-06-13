import pytest
import numpy as np
import tensorflow as tf
import larq as lq

TF_VERSION_MAJOR, TF_VERSION_MINOR, TF_VERSION_PATCH = tf.__version__.split(".", 2)
TF_VERSION_MAJOR_MINOR = float(TF_VERSION_MAJOR + "." + TF_VERSION_MINOR)


def assert_weights(weights, expected):
    for w, e in zip(weights, expected):
        np.testing.assert_allclose(np.squeeze(w), e)


@pytest.mark.skipif(
    TF_VERSION_MAJOR_MINOR > 1.13, reason="requires Tensorflow 1.13 or less"
)
class TestXavierLearingRateScaling:
    def test_xavier_scaling(self):
        dense = lq.layers.QuantDense(
            1, kernel_quantizer="ste_sign", kernel_initializer="zeros", input_shape=(1,)
        )
        model = tf.keras.models.Sequential([dense])

        model.compile(
            loss="mae",
            optimizer=lq.optimizers.XavierLearningRateScaling(
                tf.keras.optimizers.SGD(1), model
            ),
        )
        assert_weights(dense.get_weights(), [0, 0])
        model.fit(np.array([1.0]), np.array([2.0]), epochs=1, batch_size=1)
        assert_weights(dense.get_weights(), [1 / np.sqrt(1.5 / 2), 1])

    def test_invalid_usage(self):
        with pytest.raises(ValueError):
            lq.optimizers.XavierLearningRateScaling(
                tf.keras.optimizers.SGD(), "invalid"
            )
        with pytest.raises(ValueError):
            lq.optimizers.XavierLearningRateScaling("invalid", tf.keras.models.Model())

    def test_serialization(self):
        dense = lq.layers.QuantDense(10, kernel_quantizer="ste_sign", input_shape=(3,))
        model = tf.keras.models.Sequential([dense])
        ref_opt = lq.optimizers.XavierLearningRateScaling(
            tf.keras.optimizers.SGD(1), model
        )
        assert ref_opt.lr == ref_opt.optimizer.lr

        config = tf.keras.optimizers.serialize(ref_opt)
        opt = tf.keras.optimizers.deserialize(config)
        assert opt.__class__ == ref_opt.__class__
        assert opt.optimizer.__class__ == ref_opt.optimizer.__class__
        assert opt.optimizer.get_config() == ref_opt.optimizer.get_config()
        assert opt.multipliers == ref_opt.multipliers
