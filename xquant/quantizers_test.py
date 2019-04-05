import tensorflow as tf
import numpy as np

import pytest
import xquant as xq


@pytest.mark.parametrize("name", ["ste_sign", "approx_sign"])
def test_serialization(name):
    fn = xq.quantizers.get(name)
    ref_fn = getattr(xq.quantizers, name)
    assert fn == ref_fn
    config = xq.quantizers.serialize(fn)
    fn = xq.quantizers.deserialize(config)
    assert fn == ref_fn
    fn = xq.quantizers.get(ref_fn)
    assert fn == ref_fn


@pytest.mark.parametrize("name", ["ste_sign", "approx_sign"])
def test_binarization(name):
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [xq.quantizers.get(name)(x)])
    binarized_values = np.random.choice([-1, 1], size=(2, 5))
    result = f([binarized_values])[0]
    np.testing.assert_allclose(result, binarized_values)

    real_values = np.random.uniform(-2, 2, (2, 5))
    result = f([real_values])[0]
    assert not np.any(result == 0)
    assert np.all(result[result < 0] == -1)
    assert np.all(result[result >= 0] == 1)


class GradientTests(tf.test.TestCase):
    def test_ste_grad(self):
        if tf.executing_eagerly():
            return

        @np.vectorize
        def ste_grad(x):
            if np.abs(x) <= 1:
                return 1.0
            return 0.0

        x = np.random.uniform(-2, 2, (8, 3, 3, 16))
        tf_x = tf.constant(x, dtype=tf.float32)
        grad = tf.gradients(xq.quantizers.ste_sign(tf_x), tf_x)[0]
        self.assertAllClose(grad, ste_grad(x))

    def test_approx_sign_grad(self):
        if tf.executing_eagerly():
            return

        @np.vectorize
        def approx_sign_grad(x):
            if np.abs(x) <= 1:
                return 2 - 2 * np.abs(x)
            return 0.0

        x = np.random.uniform(-2, 2, (8, 3, 3, 16))
        tf_x = tf.constant(x, dtype=tf.float32)
        grad = tf.gradients(xq.quantizers.approx_sign(tf_x), tf_x)[0]
        self.assertAllClose(grad, approx_sign_grad(x))
