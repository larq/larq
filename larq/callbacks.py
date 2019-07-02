import tensorflow as tf


class HyperparameterScheduler(tf.keras.callbacks.Callback):
    """Generic hyperparameter scheduler.

    # Arguments
    schedule: a function that takes an epoch index as input
        (integer, indexed from 0) and returns a new hyperparameter as output.
    hyperparameter: str. the name of the hyperparameter to be scheduled.
    verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, hyperparameter, verbose=0):
        super(HyperparameterScheduler, self).__init__()
        self.schedule = schedule
        self.hyperparameter = hyperparameter
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, self.hyperparameter):
            raise ValueError(
                f'Optimizer must have a "{self.hyperparameter}" attribute.'
            )

        hp = getattr(self.model.optimizer, self.hyperparameter)
        try:  # new API
            hyperparameter_val = tf.keras.backend.get_value(hp)
            hyperparameter_val = self.schedule(epoch, hyperparameter_val)
        except TypeError:  # Support for old API for backward compatibility
            hyperparameter_val = self.schedule(epoch)

        tf.keras.backend.set_value(hp, hyperparameter_val)

        if self.verbose > 0:
            print(
                f"Epoch {epoch + 1}: {self.hyperparameter} changning to {tf.keras.backend.get_value(hp)}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        hp = getattr(self.model.optimizer, self.hyperparameter)
        logs[self.hyperparameter] = tf.keras.backend.get_value(hp)
