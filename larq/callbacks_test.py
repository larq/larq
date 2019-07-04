import numpy as np
import tensorflow as tf
import larq as lq

from larq import testing_utils as lq_testing_utils
from larq.callbacks import HyperparameterScheduler

from tensorflow import keras
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
    def test_hyper_parameter_scheduler(self):
        np.random.seed(1337)
        (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
            train_samples=1000, test_samples=0, input_shape=(10,), num_classes=2
        )

        y_train = keras.utils.to_categorical(y_train)

        model = lq_testing_utils.get_small_bnn_model(
            x_train.shape[1], 20, y_train.shape[1]
        )
        bop_optimizer = lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01))
        model.compile(
            loss="categorical_crossentropy",
            optimizer=bop_optimizer,
            metrics=["accuracy"],
        )

        def scheduler(x):
            return 1.0 / (1.0 + x)

        cbk_gamma_scheduler = HyperparameterScheduler(
            scheduler, hyperparameter="gamma", verbose=1
        )
        cbk_threshold_scheduler = HyperparameterScheduler(
            scheduler, hyperparameter="threshold", verbose=1
        )
        cbk_lr_scheduler = HyperparameterScheduler(
            scheduler, hyperparameter="lr", verbose=1
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
            keras.backend.get_value(model.optimizer.gamma),
            scheduler(num_epochs - 1),
            decimal=8,
        )

        np.testing.assert_almost_equal(
            keras.backend.get_value(model.optimizer.threshold),
            scheduler(num_epochs - 1),
            decimal=8,
        )

        np.testing.assert_almost_equal(
            keras.backend.get_value(model.optimizer.lr),
            scheduler(num_epochs - 1),
            decimal=8,
        )
