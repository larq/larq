import tensorflow as tf
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
