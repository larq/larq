import tensorflow as tf
import numpy as np

import pytest
import larq as lq


@pytest.mark.parametrize("name", ["ste_sign", "approx_sign"])
def test_serialization(name):
    fn = lq.quantizers.get(name)
    ref_fn = getattr(lq.quantizers, name)
    assert fn == ref_fn
    config = lq.quantizers.serialize(fn)
    fn = lq.quantizers.deserialize(config)
    assert fn == ref_fn
    fn = lq.quantizers.get(ref_fn)
    assert fn == ref_fn


def test_invalid_usage():
    with pytest.raises(ValueError):
        lq.quantizers.get(42)
    with pytest.raises(ValueError):
        lq.quantizers.get("unknown")


@pytest.mark.parametrize("name", ["ste_sign", "approx_sign"])
def test_binarization(name):
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [lq.quantizers.get(name)(x)])
    binarized_values = np.random.choice([-1, 1], size=(2, 5))
    result = f([binarized_values])[0]
    np.testing.assert_allclose(result, binarized_values)

    real_values = np.random.uniform(-2, 2, (2, 5))
    result = f([real_values])[0]
    assert not np.any(result == 0)
    assert np.all(result[result < 0] == -1)
    assert np.all(result[result >= 0] == 1)


@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_ste_grad():
    @np.vectorize
    def ste_grad(x):
        if np.abs(x) <= 1:
            return 1.0
        return 0.0

    x = np.random.uniform(-2, 2, (8, 3, 3, 16))
    tf_x = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(tf_x)
        activation = lq.quantizers.ste_sign(tf_x)
    grad = tape.gradient(activation, tf_x)
    np.testing.assert_allclose(grad.numpy(), ste_grad(x))


@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_approx_sign_grad():
    @np.vectorize
    def approx_sign_grad(x):
        if np.abs(x) <= 1:
            return 2 - 2 * np.abs(x)
        return 0.0

    x = np.random.uniform(-2, 2, (8, 3, 3, 16))
    tf_x = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(tf_x)
        activation = lq.quantizers.approx_sign(tf_x)
    grad = tape.gradient(activation, tf_x)
    np.testing.assert_allclose(grad.numpy(), approx_sign_grad(x))
