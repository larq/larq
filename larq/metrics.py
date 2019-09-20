import tensorflow as tf
from larq import utils
import numpy as np
from contextlib import contextmanager

try:
    from tensorflow.keras.metrics import Metric
except:  # TensorFlow 1.13 doesn't export this as a public API
    from tensorflow.python.keras.metrics import Metric


__all__ = ["scope", "get_training_metrics"]

_GLOBAL_TRAINING_METRICS = set()
_AVAILABLE_METRICS = {"flip_ratio", "gradient_flow"}


@contextmanager
def scope(metrics=[]):
    """A context manager to set the training metrics to be used in layers.

    !!! example
        ```python
        with larq.metrics.scope(["flip_ratio"]):
            model = tf.keras.models.Sequential(
                [larq.layers.QuantDense(3, kernel_quantizer="ste_sign", input_shape=(32,))]
            )
        model.compile(loss="mse", optimizer="sgd")
        ```

    # Arguments
    metrics: Iterable of metrics to add to layers defined inside this context.
        Currently only the `flip_ratio` metric is available.
    """
    for metric in metrics:
        if metric not in _AVAILABLE_METRICS:
            raise ValueError(
                f"Unknown training metric '{metric}'. Available metrics: {_AVAILABLE_METRICS}."
            )
    backup = _GLOBAL_TRAINING_METRICS.copy()
    _GLOBAL_TRAINING_METRICS.update(metrics)
    yield _GLOBAL_TRAINING_METRICS
    _GLOBAL_TRAINING_METRICS.clear()
    _GLOBAL_TRAINING_METRICS.update(backup)


def get_training_metrics():
    """Retrieves a live reference to the training metrics in the current scope.

    Updating and clearing training metrics using `larq.metrics.scope` is preferred,
    but `get_training_metrics` can be used to directly access them.

    !!! example
        ```python
        get_training_metrics().clear()
        get_training_metrics().add("flip_ratio")
        ```

    # Returns
    A set of training metrics in the current scope.
    """
    return _GLOBAL_TRAINING_METRICS


class LarqMetric(Metric):
    """Metric with support for both 1.13 and 1.14+"""

    def add_weight(
        self,
        name,
        shape=(),
        aggregation=tf.VariableAggregation.SUM,
        synchronization=tf.VariableSynchronization.ON_READ,
        initializer=None,
        dtype=None,
    ):
        if utils.tf_1_14_or_newer():
            return super().add_weight(
                name=name,
                shape=shape,
                aggregation=aggregation,
                synchronization=synchronization,
                initializer=initializer,
                dtype=dtype,
            )
        else:
            # Call explicitely tf.keras.layers.Layer.add_weight because TF 1.13
            # doesn't support setting a custom dtype
            return tf.keras.layers.Layer.add_weight(
                self,
                name=name,
                shape=shape,
                dtype=self._dtype if dtype is None else dtype,
                trainable=False,
                initializer=initializer,
                collections=[],
                synchronization=synchronization,
                aggregation=aggregation,
            )


@utils.register_alias("flip_ratio")
@utils.register_keras_custom_object
class FlipRatio(LarqMetric):
    """Computes the mean ration of changed values in a given tensor.

    !!! example
        ```python
        m = metrics.FlipRatio(values_shape=(2,))
        m.update_state((1, 1))  # result: 0
        m.update_state((2, 2))  # result: 1
        m.update_state((1, 2))  # result: 0.75
        print('Final result: ', m.result().numpy())  # Final result: 0.75
        ```

    # Arguments
    values_shape: Shape of the tensor for which to track changes.
    values_dtype: Data type of the tensor for which to track changes.
    name: Name of the metric.
    dtype: Data type of the moving mean.
    """

    def __init__(
        self, values_shape=(), values_dtype="int8", name="flip_ratio", dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.values_dtype = tf.as_dtype(values_dtype)
        self.values_shape = tf.TensorShape(values_shape).as_list()
        self.is_weight_metric = True
        with tf.init_scope():
            self._previous_values = self.add_weight(
                "previous_values",
                shape=values_shape,
                dtype=self.values_dtype,
                initializer=tf.keras.initializers.zeros,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
            self.total = self.add_weight(
                "total",
                initializer=tf.keras.initializers.zeros,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
            self.count = self.add_weight(
                "count",
                initializer=tf.keras.initializers.zeros,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
        self._size = np.prod(self.values_shape)

    def update_state(self, values, sample_weight=None):
        values = tf.cast(values, self.values_dtype)
        changed_values = tf.math.count_nonzero(tf.equal(self._previous_values, values))
        flip_ratio = 1 - (tf.cast(changed_values, self.dtype) / self._size)

        update_total_op = self.total.assign_add(flip_ratio * tf.sign(self.count))
        with tf.control_dependencies([update_total_op]):
            update_count_op = self.count.assign_add(1)
            with tf.control_dependencies([update_count_op]):
                return self._previous_values.assign(values)

    def result(self):
        return tf.compat.v1.div_no_nan(self.total, self.count - 1)

    def reset_states(self):
        tf.keras.backend.batch_set_value(
            [(v, 0) for v in self.variables if v is not self._previous_values]
        )

    def get_config(self):
        return {
            **super().get_config(),
            "values_shape": self.values_shape,
            "values_dtype": self.values_dtype.name,
        }


@utils.register_alias("gradient_flow")
@utils.register_keras_custom_object
class GradientFlow(LarqMetric):
    r"""Indicator of gradient mismatch and saturation as described in https://arxiv.org/pdf/1904.02823.pdf.
    Counts the ratio of activations for each neuron (i.e. pixel in a feature map) over the entire batch
    that fall in between the HardTanh clipping boundaries ([-1, 1] by default). A neuron with a score of 0
    did not have any activations in between the boundaries in this batch, and therefore received no gradients 
    (i.e. saturation). A neuron with a score of 1 was always active in between the boundaries, but therefore
    suffers from gradient mismatch. Scores are averaged over all neurons in the output.

    # Arguments
    threshold: The (absolute) clipping threshold used by the HardTanh. 1 by default.
    name: Name of the metric.
    """

    def __init__(self, threshold=1.0, name="gradient_flow", dtype=None):
        super().__init__(name=name, dtype=dtype)

        self.threshold = float(threshold)

        with tf.init_scope():
            self._threshold = tf.constant(self.threshold, dtype=self.dtype)
            self.total_value = self.add_weight(
                "total_value",
                initializer=tf.keras.initializers.zeros,
                dtype=self.dtype,
                aggregation=tf.VariableAggregation.SUM,
            )
            self.num_batches = self.add_weight(
                "num_batches",
                initializer=tf.keras.initializers.zeros,
                dtype=tf.int32,
                aggregation=tf.VariableAggregation.SUM,
            )

    def update_state(self, values):
        values = tf.cast(values, self.dtype)

        below_threshold = tf.math.count_nonzero(
            tf.math.abs(values) <= self._threshold, dtype=tf.float32
        )
        num_activations = tf.cast(tf.size(values), tf.float32)
        ratio = tf.cast(below_threshold / num_activations, self.dtype)

        update_total_op = self.total_value.assign_add(ratio)
        with tf.control_dependencies([update_total_op]):
            return self.num_batches.assign_add(1)

    def result(self):
        return tf.compat.v1.div_no_nan(
            tf.cast(self.total_value, tf.float32), tf.cast(self.num_batches, tf.float32)
        )

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
