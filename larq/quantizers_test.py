import tensorflow as tf
import numpy as np

import pytest
import larq as lq


@pytest.mark.parametrize("name", ["ste_sign", "approx_sign", "magnitude_aware_sign"])
def test_serialization(name):
    fn = lq.quantizers.get(name)
    ref_fn = getattr(lq.quantizers, name)
    assert fn == ref_fn
    assert type(fn.precision) == int
    config = lq.quantizers.serialize(fn)
    fn = lq.quantizers.deserialize(config)
    assert fn == ref_fn
    assert type(fn.precision) == int
    fn = lq.quantizers.get(ref_fn)
    assert fn == ref_fn
    assert type(fn.precision) == int


@pytest.mark.parametrize("name", ["ste_sign", "approx_sign", "magnitude_aware_sign"])
def test_serialization_as_activation(name):
    fn = tf.keras.activations.get(name)
    ref_fn = getattr(lq.quantizers, name)
    assert fn == ref_fn
    config = tf.keras.activations.serialize(fn)
    fn = tf.keras.activations.deserialize(config)
    assert fn == ref_fn
    fn = tf.keras.activations.get(ref_fn)
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


def test_ternarization_with_default_threshold():
    x = tf.keras.backend.placeholder(ndim=2)
    fn = lq.quantizers.SteTern()
    test_threshold = 0.1
    f = tf.keras.backend.function([x], [fn(x)])
    real_values = np.random.uniform(-2, 2, (2, 5))
    result = f([real_values])[0]
    assert np.all(result[result > test_threshold] == 1)
    assert np.all(result[result < -test_threshold] == -1)
    assert np.all(result[np.abs(result) < test_threshold] == 0)
    assert not np.any(result > 1)
    assert not np.any(result < -1)
    ternarized_values = np.random.choice([-1, 0, 1], size=(2, 5))
    result = f([ternarized_values])[0]
    np.testing.assert_allclose(result, ternarized_values)
    assert not np.any(result > 1)
    assert not np.any(result < -1)


def test_ternarization_with_custom_threshold():
    x = tf.keras.backend.placeholder(ndim=2)
    test_threshold = np.random.uniform(0.01, 0.8)
    fn = lq.quantizers.SteTern(threshold_value=test_threshold)
    f = tf.keras.backend.function([x], [fn(x)])
    real_values = np.random.uniform(-2, 2, (2, 5))
    result = f([real_values])[0]
    assert np.all(result[result > test_threshold] == 1)
    assert np.all(result[result < -test_threshold] == -1)
    assert np.all(result[np.abs(result) < test_threshold] == 0)
    assert not np.any(result > 1)
    assert not np.any(result < -1)
    ternarized_values = np.random.choice([-1, 0, 1], size=(2, 5))
    result = f([ternarized_values])[0]
    np.testing.assert_allclose(result, ternarized_values)
    assert not np.any(result > 1)
    assert not np.any(result < -1)


def test_ternarization_with_ternary_weight_networks():
    x = tf.keras.backend.placeholder(shape=(3, 5))
    fn = lq.quantizers.SteTern(ternary_weight_networks=True)
    f = tf.keras.backend.function([x], [fn(x)])
    real_values = np.random.uniform(-2, 2, (2, 5))
    result = f([real_values])[0]
    assert not np.any(result > 1)
    assert not np.any(result < -1)
    ternarized_values = np.random.choice([-1, 0, 1], size=(2, 5))
    result = f([ternarized_values])[0]
    np.testing.assert_allclose(result, ternarized_values)
    assert not np.any(result > 1)
    assert not np.any(result < -1)


@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_ste_grad():
    @np.vectorize
    def ste_grad(x):
        if np.abs(x) <= 1:
            return 1.0
        return 0.0

    x = np.random.uniform(-2, 2, (8, 3, 3, 16))
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
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
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        activation = lq.quantizers.approx_sign(tf_x)
    grad = tape.gradient(activation, tf_x)
    np.testing.assert_allclose(grad.numpy(), approx_sign_grad(x))


@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_magnitude_aware_sign():

    a = np.random.uniform(-2, 2, (3, 2, 2, 3))
    x = tf.Variable(a)
    with tf.GradientTape() as tape:
        y = lq.quantizers.magnitude_aware_sign(x)
    grad = tape.gradient(y, x)

    assert y.shape == x.shape

    # check sign
    np.testing.assert_allclose(tf.sign(y).numpy(), np.sign(a))

    scale_vector = [np.mean(np.reshape(np.abs(a[:, :, :, i]), [-1])) for i in range(3)]

    # check magnitude
    np.testing.assert_allclose(
        tf.reduce_mean(tf.abs(y), axis=[0, 1, 2]).numpy(),
        [np.mean(np.reshape(np.abs(a[:, :, :, i]), [-1])) for i in range(3)],
    )

    # check gradient
    np.testing.assert_allclose(
        grad.numpy(), np.where(abs(a) < 1, np.ones(a.shape) * scale_vector, 0)
    )
