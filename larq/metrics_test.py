import tensorflow as tf
from larq import metrics
from tensorflow.python.keras import keras_parameterized
import pytest


def test_scope():
    assert metrics.get_training_metrics() == set()
    with metrics.scope(["flip_ratio"]):
        assert metrics.get_training_metrics() == {"flip_ratio"}
    assert metrics.get_training_metrics() == set()
    with pytest.raises(ValueError, match=r".*unknown_metric.*"):
        with metrics.scope(["flip_ratio", "unknown_metric"]):
            pass


class FlipRatioTest(keras_parameterized.TestCase):
    def test_config(self):
        mcv = metrics.FlipRatio(
            values_shape=[3, 3], values_dtype="int16", name="mcv", dtype=tf.float16
        )
        self.assertEqual(mcv.name, "mcv")
        self.assertTrue(mcv.stateful)
        self.assertEqual(mcv.dtype, tf.float16)
        self.assertEqual(mcv.values_dtype, tf.int16)
        self.assertEqual(mcv.values_shape, [3, 3])

        mcv2 = metrics.FlipRatio.from_config(mcv.get_config())
        self.assertEqual(mcv2.name, "mcv")
        self.assertTrue(mcv2.stateful)
        self.assertEqual(mcv2.dtype, tf.float16)
        self.assertEqual(mcv2.values_dtype, tf.int16)
        self.assertEqual(mcv2.values_shape, [3, 3])

    def test_metric(self):
        mcv = metrics.FlipRatio([2])
        self.evaluate(tf.compat.v1.variables_initializer(mcv.variables))

        self.assertAllClose(0, mcv.result())
        self.assertAllClose(0, self.evaluate(mcv.total))
        self.assertAllClose(0, self.evaluate(mcv.count))

        self.evaluate(mcv.update_state([1, 1]))
        self.assertAllClose([1, 1], self.evaluate(mcv._previous_values))
        self.assertAllClose(0, self.evaluate(mcv.total))
        self.assertAllClose(1, self.evaluate(mcv.count))
        self.assertAllClose(0, mcv.result())

        self.evaluate(mcv.update_state([2, 2]))
        self.assertAllClose([2, 2], self.evaluate(mcv._previous_values))
        self.assertAllClose(1, self.evaluate(mcv.total))
        self.assertAllClose(2, self.evaluate(mcv.count))
        self.assertAllClose(1, mcv.result())

        self.evaluate(mcv.update_state([1, 2]))
        self.assertAllClose([1, 2], self.evaluate(mcv._previous_values))
        self.assertAllClose(1.5, self.evaluate(mcv.total))
        self.assertAllClose(3, self.evaluate(mcv.count))
        self.assertAllClose(1.5 / 2, mcv.result())

    @pytest.mark.skipif(tf.executing_eagerly(), reason="only applies to graph mode")
    def test_metric_in_graph_mode(self):
        mcv = metrics.FlipRatio([2])

        new_state = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2])
        update_state_op = mcv.update_state(new_state)
        metric_value = mcv.result()

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.variables_initializer(mcv.variables))

            sess.run(update_state_op, feed_dict={new_state: [1, 1]})
            sess.run(update_state_op, feed_dict={new_state: [2, 2]})
            sess.run(update_state_op, feed_dict={new_state: [1, 2]})

            previous, total, count, result = sess.run(
                [mcv._previous_values, mcv.total, mcv.count, metric_value]
            )

        self.assertAllClose([1, 2], previous)
        self.assertAllClose(1.5, total)
        self.assertAllClose(3, count)
        self.assertAllClose(1.5 / 2, result)
