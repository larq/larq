import tensorflow as tf
import numpy as np
import larq as lq
import pytest


@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_sign():
    # test random
    x = np.random.rand(100) * 100 - 0.5
    y = lq.math.sign(x)
    np.testing.assert_allclose(y.numpy(), np.array(x >= 0, dtype=np.float32) * 2 - 1)
    # test zero is mapped to one
    np.testing.assert_almost_equal(
        lq.math.sign(np.zeros(10)).numpy(), np.array(np.ones(10))
    )
