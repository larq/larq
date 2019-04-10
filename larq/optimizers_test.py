import pytest
import numpy as np
import tensorflow as tf
import larq as lq

if int(tf.__version__[0]) == 2:
    pytest.skip("This wrapper is not supported by TF 2", allow_module_level=True)


def assert_weights(weights, expected):
    for w, e in zip(weights, expected):
        np.testing.assert_allclose(np.squeeze(w), e)


def test_xavier_scaling():
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


def test_invalid_usage():
    with pytest.raises(ValueError):
        lq.optimizers.XavierLearningRateScaling(tf.keras.optimizers.SGD(), "invalid")
    with pytest.raises(ValueError):
        lq.optimizers.XavierLearningRateScaling("invalid", tf.keras.models.Model())


def test_serialization():
    dense = lq.layers.QuantDense(10, kernel_quantizer="ste_sign", input_shape=(3,))
    model = tf.keras.models.Sequential([dense])
    ref_opt = lq.optimizers.XavierLearningRateScaling(tf.keras.optimizers.SGD(1), model)
    assert ref_opt.lr == ref_opt.optimizer.lr

    config = tf.keras.optimizers.serialize(ref_opt)
    opt = tf.keras.optimizers.deserialize(config)
    assert opt.__class__ == ref_opt.__class__
    assert opt.optimizer.__class__ == ref_opt.optimizer.__class__
    assert opt.optimizer.get_config() == ref_opt.optimizer.get_config()
    assert opt.multipliers == ref_opt.multipliers
