import numpy as np

import larq as lq
import tensorflow as tf
from larq import testing_utils as lq_testing_utils
from larq.callbacks import HyperparameterScheduler
from tensorflow.python.keras import testing_utils


class LogHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batches = []
        self.epochs = []

    def _store_logs(self, storage, batch_or_epoch, logs={}):
        if [key for key in logs if "changed_quantization_ration" in key]:
            storage.append(batch_or_epoch)

    def on_batch_end(self, batch, logs={}):
        self._store_logs(self.batches, batch, logs)

    def on_epoch_end(self, epoch, logs={}):
        self._store_logs(self.epochs, epoch, logs)


class TestHyperparameterScheduler:
    def test_normal_optimizer(self):
        np.random.seed(1337)
        (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
            train_samples=1000, test_samples=0, input_shape=(10,), num_classes=2
        )

        y_train = tf.keras.utils.to_categorical(y_train)

        model = lq_testing_utils.get_small_bnn_model(
            x_train.shape[1], 20, y_train.shape[1]
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.01),
            metrics=["accuracy"],
        )

        def scheduler(x):
            return 1.0 / (1.0 + x)

        # We shouldn' t need to specify the optimizer
        test_scheduler = HyperparameterScheduler(
            schedule=scheduler, hyperparameter="lr", verbose=1,
        )

        num_epochs = 2
        model.fit(
            x_train,
            y_train,
            epochs=num_epochs,
            batch_size=16,
            callbacks=[test_scheduler],
            verbose=0,
        )

        np.testing.assert_almost_equal(
            tf.keras.backend.get_value(model.optimizer.lr),
            scheduler(num_epochs - 1),
            decimal=8,
        )

    def test_case_optimizer(self):
        np.random.seed(1337)
        (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
            train_samples=1000, test_samples=0, input_shape=(10,), num_classes=2
        )

        y_train = tf.keras.utils.to_categorical(y_train)

        model = lq_testing_utils.get_small_bnn_model(
            x_train.shape[1], 20, y_train.shape[1]
        )

        bop = lq.optimizers.Bop(threshold=1e-6, gamma=1e-3)
        adam = tf.keras.optimizers.Adam(0.01)
        case_optimizer = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, bop), default_optimizer=adam,
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer=case_optimizer,
            metrics=["accuracy"],
        )

        def scheduler(x):
            return 1.0 / (1.0 + x)

        cbk_gamma_scheduler = HyperparameterScheduler(
            schedule=scheduler,
            optimizer=model.optimizer.optimizers[0],
            hyperparameter="gamma",
            verbose=1,
        )
        cbk_threshold_scheduler = HyperparameterScheduler(
            schedule=scheduler,
            optimizer=model.optimizer.optimizers[0],
            hyperparameter="threshold",
            verbose=1,
        )
        cbk_lr_scheduler = HyperparameterScheduler(
            schedule=scheduler,
            optimizer=model.optimizer.optimizers[1],
            hyperparameter="lr",
            verbose=1,
        )

        num_epochs = 10
        model.fit(
            x_train,
            y_train,
            epochs=num_epochs,
            batch_size=16,
            callbacks=[cbk_gamma_scheduler, cbk_lr_scheduler, cbk_threshold_scheduler],
            verbose=0,
        )

        np.testing.assert_almost_equal(
            tf.keras.backend.get_value(model.optimizer.optimizers[0].gamma),
            scheduler(num_epochs - 1),
            decimal=8,
        )

        np.testing.assert_almost_equal(
            tf.keras.backend.get_value(model.optimizer.optimizers[0].threshold),
            scheduler(num_epochs - 1),
            decimal=8,
        )

        np.testing.assert_almost_equal(
            tf.keras.backend.get_value(model.optimizer.optimizers[1].lr),
            scheduler(num_epochs - 1),
            decimal=8,
        )
