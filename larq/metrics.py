"""We add metrics specific to extremely quantized networks using a
`larq.context.metrics_scope` rather than through the `metrics` parameter of
`model.compile()`, where most common metrics reside. This is because, to calculate
metrics like the `flip_ratio`, we need a layer's kernel or activation and not just the
`y_true` and `y_pred` that Keras passes to metrics defined in the usual way.
"""

import numpy as np
import tensorflow as tf

from larq import utils


@utils.register_alias("flip_ratio")
@utils.register_keras_custom_object
class FlipRatio(tf.keras.metrics.Metric):
    """Computes the mean ratio of changed values in a given tensor.

    !!! example
        ```python
        m = metrics.FlipRatio()
        m.update_state((1, 1))  # result: 0
        m.update_state((2, 2))  # result: 1
        m.update_state((1, 2))  # result: 0.75
        print('Final result: ', m.result().numpy())  # Final result: 0.75
        ```

    # Arguments
        name: Name of the metric.
        values_dtype: Data type of the tensor for which to track changes.
        dtype: Data type of the moving mean.
    """

    def __init__(self, values_dtype="int8", name="flip_ratio", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.built = False
        self.values_dtype = tf.as_dtype(values_dtype)

    def build(self, input_shape):
        self._previous_values = self.add_weight(
            "previous_values",
            shape=input_shape,
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
        self._size = tf.cast(np.prod(input_shape), self.dtype)
        self.built = True

    def update_state(self, values, sample_weight=None):
        values = tf.cast(values, self.values_dtype)

        if not self.built:
            with tf.name_scope(self.name), tf.init_scope():
                self.build(values.shape)

        unchanged_values = tf.math.count_nonzero(
            tf.equal(self._previous_values, values)
        )
        flip_ratio = 1 - (
            tf.cast(unchanged_values, self.dtype) / tf.cast(self._size, self.dtype)
        )

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
        return {**super().get_config(), "values_dtype": self.values_dtype.name}
