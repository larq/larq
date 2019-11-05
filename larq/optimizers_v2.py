import tensorflow as tf
import larq as lq

from larq import utils
from copy import deepcopy


class MaskedOptimizer:
    """An optimizer and a mask for which weights it should train.

    We provide `is_binary(var)` as example mask. For layers defined using the naming
    scheme used in the Larq Zoo, it masks which variables should be trained by a binary
    optimizer like Bop.

    !!! example
        ```python
        masked_bop = lq.optimizers.MaskedOptimizer(
            optimizer=lq.optimizers.Bop(),
            mask_fn=lq.optimizers.MaskedOptimizer.is_binary
        )
        ```
    
    # Arguments
    optimizer: a `tf.keras.optimizers.Optimizer`.
    mask_fn: a function which takes only a `tf.Variable` as input and returns True if 
        this optimizer should be used to train that variable.
    """

    def __init__(self, optimizer, mask_fn=None):
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                f"Expected `tf.keras.optimizers.Optimizer` for `optimizer` but got `{type(optimizer)}`."
            )

        self.optimizer = optimizer
        self.mask_fn = mask_fn

    @staticmethod
    def is_binary(var):
        return "/kernel" in var.name and "quant_" in var.name


@utils.register_keras_custom_object
class OptimizerGroup(tf.keras.optimizers.Optimizer):
    """A group of `MaskedOptimizer` that collectively train a model's variables.

    The set of `mask_fn` passed through the list of `MaskedOptimizer` should partition
    the model's variables, so that exactly one optimizer is responsible for training 
    each variable. The last `MaskedOptimizer` in the list may have its `mask_fn` set to
    `None`; this is the default optimizer that will train any variables not claimed by
    any previous optimizer.

    # Arguments
    masked_optimizers: a list of `MaskedOptimizer`
    """

    def __init__(self, masked_optimizers, name="Group"):
        super().__init__(name=name)

        for i, masked_opt in enumerate(masked_optimizers):
            if not isinstance(masked_opt, MaskedOptimizer):
                raise TypeError(
                    f"Expected `MaskedOptimizer` at index {i} but got `{type(masked_opt)}`."
                )

        self.masked_optimizers = masked_optimizers

    def __getattr__(self, name):
        if name == "lr":  # TODO: Which optimizer's LR should handle this?
            raise NotImplementedError()
        return super().__getattr__(name)

    def apply_gradients(self, grads_and_vars, name=None):
        opt_grads_and_vars = [[] for _ in range(len(self.masked_optimizers))]

        for grad, var in grads_and_vars:
            for i, masked_opt in enumerate(self.masked_optimizers):
                # TODO: Check partition condition?
                if masked_opt.mask_fn is None or masked_opt.mask_fn(var):
                    opt_grads_and_vars[i].append((grad, var))

        train_ops = []
        for i, masked_opt in enumerate(self.masked_optimizers):
            train_ops.append(
                masked_opt.optimizer.apply_gradients(opt_grads_and_vars[i], name=name)
            )

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
