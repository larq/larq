import tensorflow as tf


class HyperparameterScheduler(tf.keras.callbacks.Callback):
    """Generic hyperparameter scheduler.

    !!! example
        ```python
        optimizer = lq.optimizers.CaseOptimizer(
            (
                lq.optimizers.Bop.is_binary_variable,
                lq.optimizers.Bop(threshold=1e-6, gamma=1e-3),
            ),
            default_optimizer=tf.keras.optimizers.Adam(0.01),
        )
        callbacks = [
            HyperparameterScheduler(
                lambda x:0.001*(0.1**(x//30)), 'gamma', optimizer.optimizers[0]
            )
        ]
        ```
    # Arguments
    optimizer: the optimizer that contains the hyperparameter that will be scheduled.
    schedule: a function that takes an epoch index as input
        (integer, indexed from 0) and returns a new hyperparameter as output.
    hyperparameter: str. the name of the hyperparameter to be scheduled.
    verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, hyperparameter, optimizer=None, verbose=0):
        super(HyperparameterScheduler, self).__init__()
        self.optimizer = optimizer if optimizer else self.model.optimizer
        self.schedule = schedule
        self.hyperparameter = hyperparameter
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.optimizer, self.hyperparameter):
            raise ValueError(
                f'Optimizer must have a "{self.hyperparameter}" attribute.'
            )

        hp = getattr(self.optimizer, self.hyperparameter)
        try:  # new API
            hyperparameter_val = tf.keras.backend.get_value(hp)
            hyperparameter_val = self.schedule(epoch, hyperparameter_val)
        except TypeError:  # Support for old API for backward compatibility
            hyperparameter_val = self.schedule(epoch)

        tf.keras.backend.set_value(hp, hyperparameter_val)

        if self.verbose > 0:
            print(
                f"Epoch {epoch + 1}: {self.hyperparameter} changing to {tf.keras.backend.get_value(hp)}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        hp = getattr(self.optimizer, self.hyperparameter)
        logs[self.hyperparameter] = tf.keras.backend.get_value(hp)
