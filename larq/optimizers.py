"""Neural networks with extremely low-precision weights and activations, such as
Binarized Neural Networks (BNNs), usually contain a mix of low-precision weights (e.g.
1-bit) and  higher-precision weights (e.g. 8-bit, 16-bit, or 32-bit). Examples of this
include the first and last layers of image classificiation models, which have
higher-precision weights in most BNN architectures from the literature.

Training a BNN, then, consists of optimizing both low-precision and higher-precision
weights. In `larq`, we provide a mechanism to target different bit-precision variables
with different optimizers using the `CaseOptimizer` class. Modeled after the
[`tf.case`](https://www.tensorflow.org/api_docs/python/tf/case) signature,
`CaseOptimizer` accepts pairs of predicates and optimizers. A predicate, given a
variable, decides whether its optimizer should train that variable.

A `CaseOptimizer` behaves much like any other
[Keras optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers), and
once you instantiate it you can pass it to your `model.compile()` as usual. To
instantiate a `CaseOptimzer`, pass one or a list of `(predicate, optimizer)` tuples,
along with a `default` optimizer which trains any variables not claimed by another
optimizer. A variable may not be claimed by more than one optimizer's predicate.

!!! example
    ```python
    no_op_quantizer = lq.quantizers.NoOp(precision=1)
    layer = lq.layers.QuantDense(16, kernel_quantizer=no_op_quantizer)

    case_optimizer = lq.optimizers.CaseOptimizer(
        (
            lq.optimizers.Bop.is_binary_variable,  # predicate
            lq.optimizers.Bop(threshold=1e-6, gamma=1e-3),  # optimizer
        ),
        default_optimizer=tf.keras.optimizers.Adam(0.01),
    )
    ```
"""


import warnings
from copy import deepcopy
from typing import Callable, Optional, Tuple

import tensorflow as tf

import larq as lq
from larq import utils

__all__ = ["Bop", "CaseOptimizer"]


@utils.register_keras_custom_object
class CaseOptimizer(tf.keras.optimizers.Optimizer):
    """An optmizer wrapper that applies different optimizers to a subset of variables.

    An optimizer is used to train a variable iff its accompanying predicate evaluates to
    `True`.

    For each variable, at most one optimizer's predicate may evaluate to `True`. If no
    optimizer's predicate evaluates to `True` for a variable, it is trained with the
    `default_optimizer`. If a variable is claimed by no optimizers and
    `default_optimizer == None`, the variable is not trained.

    # Arguments
        predicate_optimizer_pairs: One or more `(pred, tf.keras.optimizers.Optimizer)`
            pairs, where `pred`  takes one `tf.Variable` as argument and returns `True`
            if the optimizer should be used for that variable, e.g. `pred(var) == True`.
        default_optimizer: A `tf.keras.optimizers.Optimizer` to be applied to any
            variable not claimed by any other optimizer. (Must be passed as keyword
            argument.)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        *predicate_optimizer_pairs: Tuple[
            Callable[[tf.Variable], bool], tf.keras.optimizers.Optimizer
        ],
        default_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        name: str = "optimizer_case",
    ):
        super().__init__(name=name)

        # Type checks for (predicate, optimizer) pairs
        for i, (predicate, optimizer) in enumerate(predicate_optimizer_pairs):
            if not callable(predicate):
                raise TypeError(
                    f"Expected callable predicate at `predicate_optimizer_pairs[{i}][0]` but got `{type(predicate)}`."
                )
            if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
                raise TypeError(
                    f"Expected `tf.keras.optimizers.Optimizer` at `predicate_optimizer_pairs[{i}][1]` but got `{type(optimizer)}`."
                )

        # Type check for default optimizers
        if default_optimizer is not None and not isinstance(
            default_optimizer, tf.keras.optimizers.Optimizer
        ):
            raise TypeError(
                f"Expected `tf.keras.optimizers.Optimizer` for `default_optimizer` but got `{type(default_optimizer)}`."
            )

        self.pred_opt_pairs = predicate_optimizer_pairs
        self.default = default_optimizer

        self.var_opt_mapping = None

        # List of optimizers ending in `default_optimizer`, for easier internal access
        self.optimizers = [opt for (_, opt) in self.pred_opt_pairs]

        if self.default:
            self.optimizers.append(self.default)
            self.DEFAULT_OPT_INDEX = len(self.pred_opt_pairs)

        # Track optimizers to support reloading via tf.train.Checkpoint
        for i, optimizer in enumerate(self.optimizers):
            self._track_trackable(optimizer, name=f"optimizer_{i}")

    @property
    def weights(self):
        weights = []
        for optimizer in self.optimizers:
            weights.extend(optimizer.weights)
        return weights

    @tf.keras.optimizers.Optimizer.iterations.setter
    def iterations(self, variable):
        raise NotImplementedError("CaseOptimzer does not support setting iterations.")

    def apply_gradients(self, grads_and_vars, name: Optional[str] = None, **kwargs):
        """Apply gradients to variables for each optimizer.

        On the first call to `apply_gradients()`, compute the mapping from variables to
        optimizers and cache it in the `self.var_opt_mapping` dict for serialization and
        faster access.
        """

        if self.var_opt_mapping is None:
            # Convert `grads_and_vars` to list so we can iterate multiple times over it
            grads_and_vars = list(grads_and_vars)
            self._compute_var_opt_mapping(grads_and_vars)

        # Split gradients and variables into a separate list for each optimizer
        grad_var_lists = [[] for _ in range(len(self.pred_opt_pairs) + 1)]
        for grad, var in grads_and_vars:
            if var.name in self.var_opt_mapping:
                grad_var_lists[self.var_opt_mapping[var.name]].append((grad, var))

        with tf.init_scope():
            _ = self.iterations
            # This is only necessary in TF 2.0 and older, but doesn't hurt on newer versions
            for optimizer, opt_grads_and_vars in zip(self.optimizers, grad_var_lists):
                optimizer._create_slots([v for (_, v) in grads_and_vars])

        return tf.distribute.get_replica_context().merge_call(
            self._apply_gradients, args=(grad_var_lists, name), kwargs=kwargs
        )

    def _apply_gradients(self, distribution, grad_var_lists, name, **kwargs):
        # Apply gradients to each optimizer
        with tf.name_scope(self._name):
            train_ops = [
                distribution.extended.call_for_each_replica(
                    optimizer.apply_gradients, args=(opt_grads_and_vars,), kwargs=kwargs
                )
                for optimizer, opt_grads_and_vars in zip(
                    self.optimizers, grad_var_lists
                )
            ]

            return tf.group(*train_ops, name=name or "train_with_group")

    def get_config(self):
        optimizer_configs = [opt.get_config() for (_, opt) in self.pred_opt_pairs]
        default_config = self.default.get_config()

        config = {
            "optimizer_configs": [
                {"class_name": optimizer_config["name"], "config": optimizer_config}
                for optimizer_config in optimizer_configs
            ],
            "default_config": {
                "class_name": default_config["name"],
                "config": default_config,
            },
            "var_opt_mapping": self.var_opt_mapping,  # serialized instead of `pred`s
        }
        return {**super().get_config(), **config}

    @classmethod
    def from_config(cls, original_config, custom_objects=None):
        config = deepcopy(original_config)

        case_optimizer = cls(
            *[  # `(pred, opt)` tuples
                (
                    lambda _: False,  # placeholder callable (`pred` is not serialized)
                    tf.keras.optimizers.deserialize(  # optimizer `opt`
                        opt_config, custom_objects=custom_objects
                    ),
                )
                for opt_config in config["optimizer_configs"]
            ],
            default_optimizer=tf.keras.optimizers.deserialize(
                config["default_config"], custom_objects=custom_objects
            ),
        )

        # Since we no longer have the `pred`s, we set the mapping explicitly
        case_optimizer.var_opt_mapping = config["var_opt_mapping"]

        return case_optimizer

    def _compute_var_opt_mapping(self, grads_and_vars):
        """Compute a unique mapping from variables to optimizer indices."""

        self.var_opt_mapping = {}

        for grad, var in grads_and_vars:
            num_optimizers = 0

            # Find the optimizer(s) that want to claim this variable
            for optimizer_index, (predicate, _) in enumerate(self.pred_opt_pairs):
                if predicate(var):
                    self.var_opt_mapping[var.name] = optimizer_index
                    num_optimizers += 1

            if num_optimizers > 1:
                raise ValueError(f"Variable `{var}` claimed by multiple optimizers.")
            if num_optimizers == 0:
                if self.default is not None:
                    self.var_opt_mapping[var.name] = self.DEFAULT_OPT_INDEX
                else:
                    warnings.warn(
                        f"No `default_optimizer` provided to train variable `{var}`."
                    )

        # Make sure that each optimizer touches at least one variable
        for optimizer_index, (_, optimizer) in enumerate(self.pred_opt_pairs):
            if optimizer_index not in self.var_opt_mapping.values():
                raise ValueError(
                    f"Optimizer `{optimizer}` did not claim any variables."
                )


@utils.register_keras_custom_object
class Bop(tf.keras.optimizers.Optimizer):
    """Binary optimizer (Bop).

    Bop is a latent-free optimizer for Binarized Neural Networks (BNNs) and
    Binary Weight Networks (BWN).

    Bop maintains an exponential moving average of the gradients controlled by
    `gamma`. If this average exceeds the `threshold`, a weight is flipped.

    The hyperparameter `gamma` is somewhat analogues to the learning rate in
    SGD methods: a high `gamma` results in rapid convergence but also makes
    training more noisy.

    Note that the default `threshold` is not optimal for all situations.
    Setting the threshold too high results in little learning, while setting it
    too low results in overly noisy behaviour.

    !!! warning
        The `is_binary_variable` check of this optimizer will only target variables that
        have been explicitly marked as being binary using `NoOp(precision=1)`.

    !!! example
        ```python
        no_op_quantizer = lq.quantizers.NoOp(precision=1)
        layer = lq.layers.QuantDense(16, kernel_quantizer=no_op_quantizer)

        optimizer = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
            default_optimizer=tf.keras.optimizers.Adam(0.01),  # for FP weights
        )
        ```

    # Arguments
        threshold: magnitude of average gradient signal required to flip a weight.
        gamma: the adaptivity rate.
        name: name of the optimizer.

    # References
        - [Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization](https://papers.nips.cc/paper/8971-latent-weights-do-not-exist-rethinking-binarized-neural-network-optimization)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self, threshold: float = 1e-8, gamma: float = 1e-4, name: str = "Bop", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self._set_hyper("threshold", threshold)
        self._set_hyper("gamma", gamma)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _get_decayed_hyper(self, name: str, var_dtype):
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

        m_t = m.assign_add(gamma * (grad - m))
        var_t = lq.math.sign(-tf.sign(var * m_t - threshold) * var)
        return var.assign(var_t).op

    def get_config(self):
        config = {
            "threshold": self._serialize_hyperparameter("threshold"),
            "gamma": self._serialize_hyperparameter("gamma"),
        }
        return {**super().get_config(), **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        for hyper in ("gamma", "threshold"):
            if hyper in config and isinstance(config[hyper], dict):
                config[hyper] = tf.keras.optimizers.schedules.deserialize(
                    config[hyper], custom_objects=custom_objects
                )
        return cls(**config)

    @staticmethod
    def is_binary_variable(var: tf.Variable) -> bool:
        """Returns `True` for variables with `var.precision == 1`.

        This is an example of a predictate that can be used by the `CaseOptimizer`.

        # Arguments
            var: a `tf.Variable`.
        """
        return getattr(var, "precision", 32) == 1
