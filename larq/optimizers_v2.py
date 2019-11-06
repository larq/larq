import warnings

import tensorflow as tf
import larq as lq

from larq import utils
from copy import deepcopy


@utils.register_keras_custom_object
class CaseOptimizer(tf.keras.optimizers.Optimizer):
    """A case operation for optimizers that each train a subset of a model's variables.

    An optimizer is used to train a variable iff its accompanying predicate evaluates to
    `True`.

    For each variable, at most one optimizer's predicate may evaluate to `True`. If no
    optimizer's predicate evaluates to `True` for a variable, it is trained with the
    `default` optimizer.

    # Arguments
    pred_opt_pairs: a list of `(tf.keras.optimizers.Optimzer, pred)` pairs, where `pred` 
        takes one `tf.Variable` as argument and returns `True` if the optimizer should
        train that variable, e.g. `pred(var) === True`.
    default: a `tf.keras.optimizers.Optimizer` to be applied to any variable not claimed
        by any other optimizer.
    """

    def __init__(self, pred_opt_pairs, default=None, name="optimizer_case"):
        super().__init__(name=name)

        # Type checks for (predicate, optimizer) pairs
        for i, (pred, opt) in enumerate(pred_opt_pairs):
            if not callable(pred):
                raise TypeError(
                    f"Expected callable predicate at `pred_opt_pairs[{i}][0]` but got `{type(pred)}`."
                )
            if not isinstance(opt, tf.keras.optimizers.Optimizer):
                raise TypeError(
                    f"Expected `tf.keras.optimizers.Optimizer` at `pred_opt_pairs[{i}][1]` but got `{type(opt)}`."
                )

        # Type check for default optimizers
        if default is not None and not isinstance(
            default, tf.keras.optimizers.Optimizer
        ):
            raise TypeError(
                f"Expected `tf.keras.optimizers.Optimizer` for `default` but got `{type(fallback_optimizer)}`."
            )

        self.pred_opt_pairs = pred_opt_pairs
        self.default = default

        self.var_opt_mapping = None
        self.DEFAULT_OPT_INDEX = -1

    def __getattr__(self, name):
        if name == "lr":
            raise ValueError("No single learning rate available for CaseOptimizer.")
        return super().__getattr__(name)

    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients to variables for each optimizer.
        
        On the first call to `apply_gradients()`, compute the mapping from variables to
        optimizers and cache it in the `self.var_opt_mapping` dict for serialization and
        faster access.
        """

        if self.var_opt_mapping is None:
            grads_and_vars = self._compute_var_opt_mapping(grads_and_vars)

        train_ops = [
            self._get_optimizer(var).apply_gradients([(grad, var)])
            for (grad, var) in grads_and_vars
        ]

        return tf.group(*train_ops, name="train_with_group")

    # TODO
    def get_config(self):
        raise NotImplementedError()
        fp_optimizer_config = self.fp_optimizer.get_config()
        bin_optimizer_config = self.bin_optimizer.get_config()

        config = {
            "bin_optimizer": {
                "class_name": bin_optimizer_config["name"],
                "config": bin_optimizer_config,
            },
            "fp_optimizer": {
                "class_name": fp_optimizer_config["name"],
                "config": fp_optimizer_config,
            },
        }
        return {**super().get_config(), **config}

    # TODO
    @classmethod
    def from_config(cls, original_config, custom_objects=None):
        raise NotImplementedError()
        config = deepcopy(original_config)
        return cls(
            bin_optimizer=tf.keras.optimizers.deserialize(
                config.pop("bin_optimizer"), custom_objects=custom_objects
            ),
            fp_optimizer=tf.keras.optimizers.deserialize(
                config.pop("fp_optimizer"), custom_objects=custom_objects
            ),
            **config,
        )

    def _get_optimizer(self, var):
        """Get the optimizer for a variable."""

        optimizer_index = self.var_opt_mapping[var.name]

        if optimizer_index == self.DEFAULT_OPT_INDEX:
            return self.default
        else:
            return self.pred_opt_pairs[optimizer_index][1]  # from (pred, opt) tuple

    def _compute_var_opt_mapping(self, grads_and_vars):
        """Compute a unique mapping from variables to optimizers.
        
        Also yield a new iterator for the original `grads_and_vars`, which can then be
        used by `apply_gradients()`.
        """

        self.var_opt_mapping = {}

        for grad, var in grads_and_vars:
            num_opts = 0

            for optimizer_index, (predicate, _) in enumerate(self.pred_opt_pairs):
                if predicate(var):
                    self.var_opt_mapping[var.name] = optimizer_index
                    num_opts += 1

            if num_opts > 1:
                raise ValueError(f"Variable `{var}` claimed by multiple optimizers.")
            if num_opts == 0:
                self.var_opt_mapping[var.name] = self.DEFAULT_OPT_INDEX
                if self.default is None:
                    warnings.warn(f"No `default` provided to train variable `{var}`.")

            yield ((grad, var))


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

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

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
            "name": self._name,
            "threshold": self._serialize_hyperparameter("threshold"),
            "gamma": self._serialize_hyperparameter("gamma"),
        }
        return {**super().get_config(), **config}

    @staticmethod
    def is_binary(var):
        """Returns True for binary variables named using the Larq Zoo naming scheme.
        
        # Arguments
        var: a `tf.Variable`.
        """
        return "/kernel" in var.name and "quant_" in var.name
