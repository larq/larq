import tensorflow as tf
import numpy as np


class MeanChangedValues(tf.keras.metrics.Mean):
    def __init__(
        self, values_shape, values_dtype="int8", name="mean_changed_values", dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self._values_dtype = tf.as_dtype(values_dtype).name
        self._previous_values = self.add_weight(
            "previous_values",
            shape=values_shape,
            dtype=self._values_dtype,
            initializer=tf.keras.initializers.zeros,
        )
        self._size = np.prod(tf.TensorShape(values_shape).as_list())

    def update_state(self, values, sample_weight=None):
        values = tf.cast(values, self._values_dtype)
        changed_values = tf.count_nonzero(tf.equal(self._previous_values, values))
        metric_update_op = super().update_state(changed_values / self._size)
        with tf.control_dependencies([metric_update_op]):
            return self._previous_values.assign(values)

    def reset_states(self):
        tf.keras.backend.batch_set_value(
            [(v, 0) for v in self.variables if v != self._previous_values]
        )
