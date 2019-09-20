import tensorflow as tf
from larq import metrics
from tensorflow.python.keras import keras_parameterized
import pytest
import numpy as np


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


class GradientFlowTest(keras_parameterized.TestCase):
    def test_config(self):
        mcv = metrics.GradientFlow(name="mcv", threshold=5.0, dtype="float32")
        self.assertEqual(mcv.name, "mcv")
        self.assertTrue(mcv.stateful)
        self.assertEqual(mcv.dtype, tf.float32)

        mcv2 = metrics.GradientFlow.from_config(mcv.get_config())
        self.assertEqual(mcv2.name, "mcv")
        self.assertTrue(mcv2.stateful)
        self.assertEqual(mcv2.dtype, tf.float32)
        self.assertAllClose(mcv2.threshold, 5.0)

    def test_metric(self):
        mcv = metrics.GradientFlow(threshold=1.0, dtype="float32")
        self.evaluate(tf.compat.v1.variables_initializer(mcv.variables))

        self.assertAllClose(0.0, mcv.result())
        self.assertAllClose(0, self.evaluate(mcv.total_value))

        # 2/4 all pixels in the batch receive a gradient
        self.evaluate(mcv.update_state([[-2, 0.5], [-1.1, 0.9]]))
        self.assertAllClose(0.5, self.evaluate(mcv.total_value))
        self.assertAllClose(0.5, mcv.result())

        # No gradients
        self.evaluate(mcv.update_state([[-5, 8], [2, 1.03]]))
        self.assertAllClose(0.5, self.evaluate(mcv.total_value))
        self.assertAllClose(mcv.result(), 0.25)

        # batch size 2, and 4 feature maps of 4x4 pixels
        # First sample: 1/64 pixels receives a gradient. Second sample: None.
        state = np.ones((2, 4, 4, 4)).astype(np.float32) + 0.1
        state[0, 0, 0, 0] = -1  # -1 does not count as saturation. -1.0001 does.
        self.evaluate(mcv.update_state(state))
        self.assertAllClose((1.0 / 128.0 + 0.5), self.evaluate(mcv.total_value))
        self.assertAllClose(np.mean([0.5, 0.0, 1.0 / 128.0]), mcv.result())

    @pytest.mark.skipif(tf.executing_eagerly(), reason="only applies to graph mode")
    def test_metric_in_graph_mode(self):
        mcv = metrics.GradientFlow(threshold=1.0)

        new_state = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 2, 2, 3])
        update_state_op = mcv.update_state(new_state)
        metric_value = mcv.result()

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.variables_initializer(mcv.variables))

            # Batch 1: gradient flow 15/24
            sess.run(
                update_state_op,
                feed_dict={
                    # Formatted as BxCxWxH now, but the swapaxes ensures that we get BxWxHxC.
                    new_state: np.array(
                        [
                            # Sample 1: 7/12 pixels receive gradient.
                            [
                                # Channel 1
                                [[0.5, -0.5], [-3.0, -0.9]],
                                # Channel 2
                                [[0, -0.4], [-2, 5]],
                                # Channel 3
                                [[0, -0.4], [-2, 5]],
                            ],
                            # Sample 2: 8/12 pixels receive gradient.
                            [
                                # Channel 1,
                                [[-0.5, -0.5], [-0.02, 0.2]],
                                # Channel 2
                                [[0, -0.4], [-2, 5]],
                                # Channel 3
                                [[0, -0.4], [-2, 5]],
                            ],
                        ]
                    )
                    .swapaxes(3, 1)
                    .swapaxes(1, 2)
                    .astype(np.float32)
                },
            )

            # Batch 2: gradient flow 16/24
            sess.run(
                update_state_op,
                feed_dict={
                    # Formatted as BxCxWxH now, but the swapaxes ensures that we get BxWxHxC.
                    new_state: np.array(
                        [
                            # Sample 1: 7/12 receive gradient
                            [
                                # Channel 1
                                [[-3.0, -0.5], [-3.0, 0.5]],
                                # Channel 2
                                [[0, -0.4], [-5, 3]],
                                # Channel 3
                                [[0, -0.4], [-0.5, 3]],
                            ],
                            # Sample 2: 9/12 receive gradient
                            [
                                # Channel 1
                                [[5.0, -0.5], [-0.02, -0.5]],
                                # Channel 2
                                [[0, -0.4], [-0.8, 5]],
                                # Channel 3
                                [[0, -0.4], [-0.9, 5]],
                            ],
                        ]
                    )
                    .swapaxes(3, 1)
                    .swapaxes(1, 2)
                    .astype(np.float32)
                },
            )

            total_value, result = sess.run([mcv.total_value, metric_value])

        # 15/24 + 16/24 is 31/24, divided by two batches is 31/48
        self.assertAllClose(31.0 / 24.0, total_value)
        self.assertAllClose(31.0 / 48.0, result)
