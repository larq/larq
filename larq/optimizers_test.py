import pytest
import numpy as np
import tensorflow as tf
import larq as lq
import distutils.version

from larq import testing_utils as lq_testing_utils
from tensorflow import keras
from tensorflow.python.keras import testing_utils


class AssertLRCallback(keras.callbacks.Callback):
    def __init__(self, schedule):
        self.schedule = schedule
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        learning_rate = keras.backend.get_value(self.model.optimizer.lr)
        np.testing.assert_allclose(learning_rate, self.schedule(epoch))


def assert_weights(weights, expected):
    for w, e in zip(weights, expected):
        np.testing.assert_allclose(np.squeeze(w), e)


def _test_optimizer(optimizer, target=0.75, test_kernels_are_binary=True):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=1000, test_samples=0, input_shape=(10,), num_classes=2
    )
    y_train = keras.utils.to_categorical(y_train)

    model = lq_testing_utils.get_small_bnn_model(x_train.shape[1], 20, y_train.shape[1])
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

    # Note that when kernels are treated as latent weights they need not be
    # binary (see https://arxiv.org/abs/1906.02107 for further discussion)
    if test_kernels_are_binary:
        for layer in model.layers:
            if "quant" in layer.name:
                assert np.all(np.isin(layer.get_weights(), [-1, 1]))

    assert history.history["acc"][-1] >= target


def _test_serialization(optimizer):
    config = keras.optimizers.serialize(optimizer)
    optim = keras.optimizers.deserialize(config)
    new_config = keras.optimizers.serialize(optim)
    assert config == new_config


@pytest.mark.skipif(
    distutils.version.LooseVersion(tf.__version__)
    >= distutils.version.LooseVersion("1.14"),
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
        _test_optimizer(
            lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01)),
            test_kernels_are_binary=True,
        )

    def test_bop_lr_scheduler(self):
        (x_train, y_train), _ = testing_utils.get_test_data(
            train_samples=100, test_samples=0, input_shape=(10,), num_classes=2
        )
        y_train = keras.utils.to_categorical(y_train)

        model = lq_testing_utils.get_small_bnn_model(
            x_train.shape[1], 10, y_train.shape[1]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01)),
        )

        model.fit(
            x_train,
            y_train,
            epochs=4,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1 / (1 + epoch)),
                AssertLRCallback(lambda epoch: 1 / (1 + epoch)),
            ],
            batch_size=8,
            verbose=0,
        )

    def test_bop_serialization(self):
        _test_serialization(
            lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01))
        )
