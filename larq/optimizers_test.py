import pytest
import numpy as np
import tensorflow as tf
import larq as lq

from larq import optimizers

from tensorflow.python import keras
from tensorflow.python.keras import testing_utils


def assert_weights(weights, expected):
    for w, e in zip(weights, expected):
        np.testing.assert_allclose(np.squeeze(w), e)


def _get_bnn_model(input_dim, num_hidden, output_dim):
    model = keras.models.Sequential()
    model.add(
        lq.layers.QuantDense(
            units=num_hidden,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            activation="relu",
            input_shape=(input_dim,),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        lq.layers.QuantDense(
            units=output_dim,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            input_quantizer="ste_sign",
            activation="softmax",
        )
    )
    return model


def _test_optimizer(optimizer, target=0.75):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=1000, test_samples=200, input_shape=(10,), num_classes=2
    )
    y_train = keras.utils.to_categorical(y_train)

    model = _get_bnn_model(x_train.shape[1], 20, y_train.shape[1])
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    np.testing.assert_equal(keras.backend.get_value(model.optimizer.iterations), 0)

    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    np.testing.assert_equal(
        keras.backend.get_value(model.optimizer.iterations), 126
    )  # 63 steps per epoch

    assert history.history["acc"][-1] >= target


def _test_serialization(optimizer):
    config = keras.optimizers.serialize(optimizer)
    optim = keras.optimizers.deserialize(config)
    new_config = keras.optimizers.serialize(optim)
    assert config == new_config


@pytest.mark.skipif(
    lq.utils.get_tf_version_major_minor_float() > 1.13,
    reason="current implementation requires Tensorflow 1.13 or less",
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


class TestBopOptimizer:
    def test_bop_accuracy(self):
        _test_optimizer(lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01)))

    def test_bop_serialization(self):
        _test_serialization(
            lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01))
        )
