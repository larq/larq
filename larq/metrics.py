import tensorflow as tf
from larq import utils
import numpy as np


class MeanChangedValues(tf.keras.metrics.Mean):
    """Computes the mean ration of changed values in a given tensor.

    !!! example
        ```python
        m = metrics.MeanChangedValues()
        m.update_state(1)
        m.update_state(2)
        m.update_state(2)
        print('Final result: ', m.result().numpy())  # Final result: 0.5
        ```

    # Arguments
    values_shape: Shape of the tensor for which to track changes.
    values_dtype: Data type of the tensor for which to track changes.
    name: Name of the metric.
    dtype: Data type of the moving mean.
    """

    def __init__(
        self,
        values_shape=(),
        values_dtype="int8",
        name="mean_changed_values",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
        self.values_dtype = tf.as_dtype(values_dtype)
        self.values_shape = tf.TensorShape(values_shape).as_list()
        with tf.name_scope(name):
            self._previous_values = self.add_weight(
                "previous_values",
                shape=values_shape,
                dtype=self.values_dtype,
                initializer=tf.keras.initializers.zeros,
            )
        self._size = np.prod(self.values_shape)
        self._built = False

    def update_state(self, values, sample_weight=None):
        values = tf.cast(values, self.values_dtype)
        if not self._built:
            self._built = True
            return self._previous_values.assign(values)
        changed_values = tf.math.count_nonzero(tf.equal(self._previous_values, values))
        metric_update_op = super(MeanChangedValues, self).update_state(
            1 - (tf.cast(changed_values, self.dtype) / self._size)
        )
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
