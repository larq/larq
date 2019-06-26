import tensorflow as tf
import numpy as np
import larq as lq


def test_sign():
    # test random
    x = np.random.rand(100) * 100 - 0.5
    y = lq.math.sign(x)
    np.testing.assert_allclose(
        tf.keras.backend.get_value(y), np.array(x >= 0, dtype=np.float32) * 2 - 1
    )
    # test zero is mapped to one
    np.testing.assert_almost_equal(
        tf.keras.backend.get_value(lq.math.sign(np.zeros(10))), np.array(np.ones(10))
    )
