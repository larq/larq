import tensorflow as tf
from larq import utils
import numpy as np

try:
    from tensorflow.keras.metrics import Metric
except:  # TensorFlow 1.13 doesn't export this as a public API
    from tensorflow.python.keras.metrics import Metric


class FlipRatio(Metric):
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
        with tf.init_scope():
            self._previous_values = self.add_weight(
                "previous_values",
                shape=values_shape,
                dtype=self.values_dtype,
                initializer=tf.keras.initializers.zeros,
            )
            self.total = self.add_weight(
                "total", initializer=tf.keras.initializers.zeros
            )
            self.count = self.add_weight(
                "count", initializer=tf.keras.initializers.zeros
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
            [(v, 0) for v in self.variables if v != self._previous_values]
        )

    def get_config(self):
        return {
            **super().get_config(),
            "values_shape": self.values_shape,
            "values_dtype": self.values_dtype.name,
        }

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
