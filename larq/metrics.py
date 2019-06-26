import tensorflow as tf
from larq import utils
import numpy as np


class MeanChangedValues(tf.keras.metrics.Mean):
    def __init__(
        self,
        values_shape=None,
        values_dtype="int8",
        name="mean_changed_values",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
        self.values_dtype = tf.as_dtype(values_dtype)
        self.values_shape = tf.TensorShape(values_shape).as_list()
        self._previous_values = self.add_weight(
            "previous_values",
            shape=values_shape,
            dtype=self.values_dtype,
            initializer=tf.keras.initializers.zeros,
        )
        self._size = np.prod(self.values_shape)

    def update_state(self, values, sample_weight=None):
        values = tf.cast(values, self.values_dtype)
        changed_values = tf.math.count_nonzero(tf.equal(self._previous_values, values))
        metric_update_op = super().update_state(changed_values / self._size)
        with tf.control_dependencies([metric_update_op]):
            return self._previous_values.assign(values)

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
