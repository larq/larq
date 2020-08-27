import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import testing_utils

import larq as lq
from larq import testing_utils as lq_testing_utils


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


class TestCaseOptimizer:
    def test_type_check_predicate(self):
        with pytest.raises(TypeError):
            lq.optimizers.CaseOptimizer((False, lq.optimizers.Bop()))

    def test_type_check_optimizer(self):
        with pytest.raises(TypeError):
            lq.optimizers.CaseOptimizer((lq.optimizers.Bop.is_binary_variable, False))

    def test_type_check_default(self):
        with pytest.raises(TypeError):
            lq.optimizers.CaseOptimizer(
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
                (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
            )

            # Simple MNIST model
            mnist = tf.keras.datasets.mnist
            (train_images, train_labels), _ = mnist.load_data()
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    lq.layers.QuantDense(
                        64,
                        input_quantizer="ste_sign",
                        kernel_quantizer=lq.quantizers.NoOp(precision=1),
                        activation="relu",
                    ),
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

    def test_wrong_predicate(self):
        """Make sure we throw when an optimizer does not claim variables."""

        with pytest.raises(ValueError):
            naughty_case_opt = lq.optimizers.CaseOptimizer(
                (lambda var: False, lq.optimizers.Bop()),
                default_optimizer=tf.keras.optimizers.Adam(0.01),
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

    def test_weights(self):
        (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                lq.layers.QuantDense(
                    64,
                    input_quantizer="ste_sign",
                    kernel_quantizer=lq.quantizers.NoOp(precision=1),
                    activation="relu",
                ),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=lq.optimizers.CaseOptimizer(
                (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
                default_optimizer=tf.keras.optimizers.Adam(0.01),
            ),
        )
        model.fit(train_images[:1], train_labels[:1], epochs=1)

        opt_weights = model.optimizer.weights
        assert len(opt_weights) != 0
        checked_weights = 0
        for opt in model.optimizer.optimizers:
            for weight in opt.weights:
                assert weight is opt_weights[checked_weights]
                checked_weights += 1
        assert checked_weights == len(opt_weights)

    @pytest.mark.usefixtures("eager_mode")
    def test_checkpoint(self, tmp_path):
        # Build and run a simple model.
        var = tf.Variable([2.0])
        opt = tf.keras.optimizers.SGD(1.0, momentum=1.0)
        opt = lq.optimizers.CaseOptimizer((lambda var: True, opt))
        opt.minimize(lambda: var + 1.0, var_list=[var])
        slot_var = opt.optimizers[0].get_slot(var, "momentum")
        slot_value = slot_var.numpy().item()

        # Save a checkpoint.
        checkpoint = tf.train.Checkpoint(optimizer=opt, var=var)
        save_path = checkpoint.save(tmp_path / "ckpt")

        # Run model again.
        opt.minimize(lambda: var + 1.0, var_list=[var])
        assert slot_var.numpy().item() != slot_value

        # Load checkpoint and ensure loss scale is back to its original value.
        status = checkpoint.restore(save_path)
        status.assert_consumed()
        status.run_restore_ops()
        assert slot_var.numpy().item() == slot_value


class TestBopOptimizer:
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

    @pytest.mark.usefixtures("distribute_scope")
    def test_mixed_precision(self):
        opt = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
            default_optimizer=tf.keras.optimizers.Adam(0.01),
        )
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
        _test_optimizer(opt, test_kernels_are_binary=True)

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

    def test_bop_serialization(self):
        _test_serialization(
            lq.optimizers.CaseOptimizer(
                (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
                default_optimizer=tf.keras.optimizers.Adam(0.01),
            ),
        )

    @pytest.mark.parametrize(
        "hyper",
        [5e-4, tf.keras.optimizers.schedules.PolynomialDecay(5e-4, 100)],
    )
    def test_bop_serialization_schedule(self, hyper):
        bop = lq.optimizers.Bop(
            gamma=hyper,
            threshold=hyper,
        )
        new_bop = lq.optimizers.Bop.from_config(bop.get_config())
        assert isinstance(new_bop._get_hyper("gamma"), type(bop._get_hyper("gamma")))
        assert isinstance(
            new_bop._get_hyper("threshold"), type(bop._get_hyper("threshold"))
        )
