import tensorflow as tf
import larq as lq

from larq import utils
from copy import deepcopy


@utils.register_keras_custom_object
class Bop(tf.keras.optimizers.Optimizer):
    """Binary optimizer (Bop).

    Bop is a latent-free optimizer for Binarized Neural Networks (BNNs) and
    Binary Weight Networks (BWN).

    Bop maintains an exponential moving average of the gradients controlled by
    `gamma`. If this average exceeds the `threshold`, a weight is flipped.
    Additionally, Bop accepts a regular optimizer that is applied to the
    non-binary weights in the network.

    The hyperparameter `gamma` is somewhat analogues to the learning rate in
    SGD methods: a high `gamma` results in rapid convergence but also makes
    training more noisy.

    Note that the default `threshold` is not optimal for all situations.
    Setting the threshold too high results in little learning, while setting it
    too low results in overly noisy behaviour.

    !!! example
        ```python
        optimizer = lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01))
        ```

    # Arguments
    fp_optimizer: a `tf.keras.optimizers.Optimizer`.
    threshold: determines to whether to flip each weight.
    gamma: the adaptivity rate.
    name: name of the optimizer.

    # References
    - [Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization](https://arxiv.org/abs/1906.02107)
    """

    def __init__(self, fp_optimizer, threshold=1e-7, gamma=1e-2, name="Bop", **kwargs):
        super().__init__(name=name, **kwargs)

        if not isinstance(fp_optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                f"Expected tf.keras.optimizers.Optimizer, received {type(fp_optimizer)}."
            )

        self.fp_optimizer = fp_optimizer
        self._set_hyper("threshold", threshold)
        self._set_hyper("gamma", gamma)

    def _create_slots(self, var_list):
        for var in var_list:
            if self.is_binary(var):
                self.add_slot(var, "m")

    def apply_gradients(self, grads_and_vars, name=None):
        bin_grads_and_vars, fp_grads_and_vars = [], []
        for grad, var in grads_and_vars:
            if self.is_binary(var):
                bin_grads_and_vars.append((grad, var))
            else:
                fp_grads_and_vars.append((grad, var))

        bin_train_op = super().apply_gradients(bin_grads_and_vars, name=name)

        fp_train_op = self.fp_optimizer.apply_gradients(fp_grads_and_vars, name=name)
        return tf.group(bin_train_op, fp_train_op, name="train_with_bop")

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError()

    def __getattr__(self, name):
        if name == "lr":
            return self.fp_optimizer.lr
        return super().__getattr__(name)

    def _get_decayed_hyper(self, name, var_dtype):
        hyper = self._get_hyper(name, var_dtype)
        if isinstance(hyper, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            hyper = tf.cast(hyper(local_step), var_dtype)
        return hyper

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        gamma = self._get_decayed_hyper("gamma", var_dtype)
        threshold = self._get_decayed_hyper("threshold", var_dtype)
        m = self.get_slot(var, "m")

        m_t = tf.compat.v1.assign(
            m, (1 - gamma) * m + gamma * grad, use_locking=self._use_locking
        )
        var_t = lq.math.sign(-tf.sign(var * m_t - threshold) * var)
        return tf.compat.v1.assign(var, var_t, use_locking=self._use_locking).op

    @staticmethod
    def is_binary(var):
        return "/kernel" in var.name and "quant_" in var.name

    def get_config(self):
        fp_optimizer_config = self.fp_optimizer.get_config()
        config = {
            "threshold": self._serialize_hyperparameter("threshold"),
            "gamma": self._serialize_hyperparameter("gamma"),
            "fp_optimizer": {
                "class_name": fp_optimizer_config["name"],
                "config": fp_optimizer_config,
            },
        }
        return {**super().get_config(), **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        new_config = deepcopy(config)
        fp_optimizer = tf.keras.optimizers.deserialize(
            new_config["fp_optimizer"], custom_objects=custom_objects
        )
        new_config.pop("fp_optimizer", None)
        return cls(fp_optimizer, **new_config)
