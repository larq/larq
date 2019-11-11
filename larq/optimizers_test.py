import pytest
import numpy as np
import tensorflow as tf
import larq as lq

from larq import utils, testing_utils as lq_testing_utils
from tensorflow import keras
from tensorflow.python.keras import testing_utils


def _assert_weights(weights, expected):
    for w, e in zip(weights, expected):
        np.testing.assert_allclose(np.squeeze(w), e)


def _test_optimizer(
    optimizer, target=0.75, test_kernels_are_binary=True, trainable_bn=True
):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=1000, test_samples=0, input_shape=(10,), num_classes=2
    )
    y_train = keras.utils.to_categorical(y_train)

    model = lq_testing_utils.get_small_bnn_model(
        x_train.shape[1], 20, y_train.shape[1], trainable_bn=trainable_bn
    )
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

    initial_vars = [tf.keras.backend.get_value(w) for w in model.trainable_weights]

    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

    trained_vars = [tf.keras.backend.get_value(w) for w in model.trainable_weights]

    # check all trainable variables have actually been updated
    for v0, v1 in zip(initial_vars, trained_vars):
        assert not np.all(v0 == v1)

    # Note that when kernels are treated as latent weights they need not be
    # binary (see https://arxiv.org/abs/1906.02107 for further discussion)
    if test_kernels_are_binary:
        for layer in model.layers:
            if "quant" in layer.name:
                for weight in layer.trainable_weights:
                    assert np.all(np.isin(tf.keras.backend.get_value(weight), [-1, 1]))

    assert history.history["acc"][-1] >= target


def _test_serialization(optimizer):
    config = keras.optimizers.serialize(optimizer)
    optim = keras.optimizers.deserialize(config)
    new_config = keras.optimizers.serialize(optim)
    assert config == new_config


@pytest.mark.skipif(
    not utils.tf_1_14_or_newer(),
    reason="Only supporting CaseOptimizer in TensorFlow >= 1.14.",
)
class TestCaseOptimizer:
    def test_type_check_predicate(self):
        with pytest.raises(TypeError):
            naughty_case_opt = lq.optimizers.CaseOptimizer((False, lq.optimizers.Bop()))

    def test_type_check_optimizer(self):
        with pytest.raises(TypeError):
            naughty_case_opt = lq.optimizers.CaseOptimizer(
                (lq.optimizers.Bop.is_binary_variable, False)
            )

    def test_type_check_default(self):
        with pytest.raises(TypeError):
            naughty_case_opt = lq.optimizers.CaseOptimizer(
                (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
                default_optimizer=False,
            )

    def test_overlapping_predicates(self):
        with pytest.raises(ValueError):
            naughty_case_opt = lq.optimizers.CaseOptimizer(
                (lambda var: True, lq.optimizers.Bop()),
                (lambda var: True, lq.optimizers.Bop()),
            )
            _test_optimizer(naughty_case_opt)

    def test_missing_default(self):
        with pytest.warns(Warning):
            naughty_case_opt = lq.optimizers.CaseOptimizer(
                (lambda var: False, lq.optimizers.Bop())
            )

            # Simple MNIST model
            mnist = tf.keras.datasets.mnist
            (train_images, train_labels), _ = mnist.load_data()
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=naughty_case_opt,
                metrics=["acc"],
            )

            # Should raise on first call to apply_gradients()
            model.fit(train_images[:1], train_labels[:1], epochs=1)


@pytest.mark.skipif(
    utils.tf_1_14_or_newer(),
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
        _assert_weights(dense.get_weights(), [0, 0])
        model.fit(np.array([1.0]), np.array([2.0]), epochs=1, batch_size=1)
        _assert_weights(dense.get_weights(), [1 / np.sqrt(1.5 / 2), 1])

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
    @pytest.mark.skipif(
        utils.tf_1_14_or_newer() is False,
        reason="Only supporting CaseOptimizer in TensorFlow >= 1.14",
    )
    def test_bop_accuracy(self):
        _test_optimizer(
            lq.optimizers.CaseOptimizer(
                (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
                default_optimizer=tf.keras.optimizers.Adam(0.01),
            ),
            test_kernels_are_binary=True,
        )
        # test optimizer on model with only binary trainable vars (low accuracy)
        _test_optimizer(
            lq.optimizers.CaseOptimizer(
                (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
                default_optimizer=tf.keras.optimizers.Adam(0.01),
            ),
            test_kernels_are_binary=True,
            trainable_bn=False,
            target=0,
        )

    @pytest.mark.skipif(
        utils.tf_1_14_or_newer() is False,
        reason="tf.keras.optimizers.schedules only exists in TensorFlow 1.14",
    )
    def test_bop_tf_1_14_schedules(self):
        _test_optimizer(
            lq.optimizers.CaseOptimizer(
                (
                    lq.optimizers.Bop.is_binary_variable,
                    lq.optimizers.Bop(
                        threshold=tf.keras.optimizers.schedules.InverseTimeDecay(
                            3.0, decay_steps=1.0, decay_rate=0.5
                        ),
                        gamma=tf.keras.optimizers.schedules.InverseTimeDecay(
                            3.0, decay_steps=1.0, decay_rate=0.5
                        ),
                    ),
                ),
                default_optimizer=tf.keras.optimizers.Adam(0.01),
            ),
            test_kernels_are_binary=True,
        )

    @pytest.mark.skipif(
        utils.tf_1_14_or_newer() is False,
        reason="Only supporting CaseOptimizer in TensorFlow >= 1.14",
    )
    def test_bop_serialization(self):
        _test_serialization(
            lq.optimizers.CaseOptimizer(
                (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
                default_optimizer=tf.keras.optimizers.Adam(0.01),
            ),
        )
