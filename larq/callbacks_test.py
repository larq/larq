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
    pass
