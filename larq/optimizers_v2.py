import tensorflow as tf
import larq as lq

from larq import utils
from copy import deepcopy


@utils.register_keras_custom_object
class BNNOptimizerDuo(tf.keras.optimizers.Optimizer):
    """Group of a full-precision and binary optimizer.

    #TODO: Do we want to abstract this to `n` optimizers, not specifically one fp and
    one bin? (I personally prefer the user-side simplicity of this current approach.)
    """

    def __init__(self, bin_optimizer, fp_optimizer, name="Group"):
        super().__init__(name=name)  # TODO: Do we need to pass **kwargs?

        type_err_msg = "Expected tf.keras.optimizers.Optimizer for `{}`, received `{}.`"
        if not isinstance(bin_optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(type_err_msg.format("bin_optimizer", type(bin_optimizer)))
        if not isinstance(fp_optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(type_err_msg.format("fp_optimizer", type(bin_optimizer)))

        self.bin_optimizer = bin_optimizer
        self.fp_optimizer = fp_optimizer

    def apply_gradients(self, grads_and_vars, name=None):
        bin_grads_and_vars, fp_grads_and_vars = [], []

        for grad, var in grads_and_vars:
            if self.is_binary(var):
                bin_grads_and_vars.append((grad, var))
            else:
                fp_grads_and_vars.append((grad, var))

        bin_train_op = self.bin_optimizer.apply_gradients(bin_grads_and_vars, name=name)
        fp_train_op = self.fp_optimizer.apply_gradients(fp_grads_and_vars, name=name)

        return tf.group(bin_train_op, fp_train_op, name="train_with_group")

    def __getattr__(self, name):
        if name == "lr":
            return self.fp_optimizer.lr
        return super().__getattr__(name)  # TODO is this still robust enough?

    # TODO: Not sure if this needs to be here or forwarded to Bop.
    def _get_decayed_hyper(self, name, var_dtype):
        hyper = self._get_hyper(name, var_dtype)
        if isinstance(hyper, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            hyper = tf.cast(hyper(local_step), var_dtype)
        return hyper

    @staticmethod
    def is_binary(var):
        return "/kernel" in var.name and "quant_" in var.name

    def get_config(self):
        fp_optimizer_config = self.fp_optimizer.get_config()
        bin_optimizer_config = self.bin_optimizer.get_config()

        config = {
            "fp_optimizer": {
                "class_name": fp_optimizer_config["name"],
                "config": fp_optimizer_config,
            },
            "bin_optimizer": {
                "class_name": bin_optimizer_config["name"],
                "config": bin_optimizer_config,
            },
        }
        return {**super().get_config(), **config}

    # TODO: Fix this? What does it do?
    @classmethod
    def from_config(cls, config, custom_objects=None):
        new_config = deepcopy(config)
        fp_optimizer = tf.keras.optimizers.deserialize(
            new_config["fp_optimizer"], custom_objects=custom_objects
        )
        new_config.pop("fp_optimizer", None)
        return cls(fp_optimizer, **new_config)


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

    #TODO: Add OptimizerGroup explanation and update example

    !!! example
        ```python
        optimizer = lq.optimizers.Bop(fp_optimizer=tf.keras.optimizers.Adam(0.01))
        ```

    # Arguments
    threshold: determines to whether to flip each weight.
    gamma: the adaptivity rate.
    name: name of the optimizer.

    # References
    - [Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization](https://arxiv.org/abs/1906.02107)
    """

    def __init__(self, threshold=1e-7, gamma=1e-2, name="Bop", **kwargs):
        super().__init__(name=name, **kwargs)

        self._set_hyper("threshold", threshold)
        self._set_hyper("gamma", gamma)

    # TODO: Why do we need to do this here? (It gets called in apply_gradients().)
    # TODO: Can we remove the `is_binary` check? Will this never get called with non-
    # binary variables anyway?
    def _create_slots(self, var_list):
        for var in var_list:
            if BNNOptimizerDuo.is_binary(var):
                self.add_slot(var, "m")

    # TODO: Not sure if this needs to be here or in OptimizerGroup.
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

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError()

    def get_config(self):
        config = {
            "name": self._name,  # TODO: Check if this is correct
            "threshold": self._serialize_hyperparameter("threshold"),
            "gamma": self._serialize_hyperparameter("gamma"),
        }
        return {**super().get_config(), **config}
