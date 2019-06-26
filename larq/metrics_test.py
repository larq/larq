import tensorflow as tf
from tensorflow.python.framework import test_util
from larq import metrics


class MeanChangedValuesTest(tf.test.TestCase):
    def test_config(self):
        mcv = metrics.MeanChangedValues(
            values_shape=[3, 3], values_dtype="int16", name="mcv", dtype=tf.float16
        )
        self.assertEqual(mcv.name, "mcv")
        self.assertTrue(mcv.stateful)
        self.assertEqual(mcv.dtype, tf.float16)
        self.assertEqual(mcv.values_dtype, tf.int16)
        self.assertEqual(mcv.values_shape, [3, 3])

        mcv2 = metrics.MeanChangedValues.from_config(mcv.get_config())
        self.assertEqual(mcv2.name, "mcv")
        self.assertTrue(mcv2.stateful)
        self.assertEqual(mcv2.dtype, tf.float16)
        self.assertEqual(mcv2.values_dtype, tf.int16)
        self.assertEqual(mcv2.values_shape, [3, 3])

    @test_util.run_in_graph_and_eager_modes
    def test_metric(self):
        mcv = metrics.MeanChangedValues([2])
        self.evaluate(tf.compat.v1.variables_initializer(mcv.variables))

        self.assertAllClose(0, mcv.result())
        self.assertAllClose(0, self.evaluate(mcv.total))
        self.assertAllClose(0, self.evaluate(mcv.count))

        # Ignore the first update since it is used to setup the previous_values
        self.evaluate(mcv.update_state([1, 1]))
        self.assertAllClose([1, 1], self.evaluate(mcv._previous_values))
        self.assertAllClose(0, self.evaluate(mcv.total))
        self.assertAllClose(0, self.evaluate(mcv.count))
        self.assertAllClose(0, mcv.result())

        self.evaluate(mcv.update_state([2, 2]))
        self.assertAllClose([2, 2], self.evaluate(mcv._previous_values))
        self.assertAllClose(1, self.evaluate(mcv.total))
        self.assertAllClose(1, self.evaluate(mcv.count))
        self.assertAllClose(1, mcv.result())

        self.evaluate(mcv.update_state([1, 2]))
        self.assertAllClose([1, 2], self.evaluate(mcv._previous_values))
        self.assertAllClose(1.5, self.evaluate(mcv.total))
        self.assertAllClose(2, self.evaluate(mcv.count))
        self.assertAllClose(1.5 / 2, mcv.result())
