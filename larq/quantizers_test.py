import tensorflow as tf
import numpy as np

import pytest
import larq as lq


@pytest.mark.parametrize("module", [lq.quantizers, tf.keras.activations])
@pytest.mark.parametrize(
    "name",
    ["ste_sign", "approx_sign", "magnitude_aware_sign", "swish_sign", "ste_tern"],
)
def test_serialization(module, name):
    fn = module.get(name)
    ref_fn = getattr(lq.quantizers, name)
    assert fn == ref_fn
    assert type(fn.precision) == int
    config = module.serialize(fn)
    fn = module.deserialize(config)
    assert fn == ref_fn
    assert type(fn.precision) == int
    fn = module.get(ref_fn)
    assert fn == ref_fn
    assert type(fn.precision) == int


@pytest.mark.parametrize(
    "ref_fn",
    [
        lq.quantizers.SteSign(),
        lq.quantizers.MagnitudeAwareSign(),
        lq.quantizers.SwishSign(),
        lq.quantizers.SteTern(),
    ],
)
def test_serialization_cls(ref_fn):
    assert type(ref_fn.precision) == int
    config = lq.quantizers.serialize(ref_fn)
    fn = lq.quantizers.deserialize(config)
    assert fn.__class__ == ref_fn.__class__


def test_invalid_usage():
    with pytest.raises(ValueError):
        lq.quantizers.get(42)
    with pytest.raises(ValueError):
        lq.quantizers.get("unknown")


@pytest.mark.parametrize("name", ["ste_sign", "approx_sign", "swish_sign"])
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


@pytest.mark.parametrize("name", ["ste_heaviside"])
def test_and_binarization(name):
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [lq.quantizers.get(name)(x)])

    real_values = np.random.uniform(-2, 2, (2, 5))
    result = f([real_values])[0]
    assert np.all(result[result <= 0] == 0)
    assert np.all(result[result > 0] == 1)


@pytest.mark.parametrize("fn", [lq.quantizers.SteTern(), lq.quantizers.ste_tern])
def test_ternarization_with_default_threshold(fn):
    x = tf.keras.backend.placeholder(ndim=2)
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


@pytest.mark.parametrize(
    "fn", [lq.quantizers.ste_sign, lq.quantizers.ste_tern, lq.quantizers.ste_heaviside]
)
@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_identity_ste_grad(fn):
    x = np.random.uniform(-5, 5, (8, 3, 3, 16))
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        activation = fn(tf_x, clip_value=None)
    grad = tape.gradient(activation, tf_x)
    np.testing.assert_allclose(grad.numpy(), np.ones_like(x))


@pytest.mark.parametrize(
    "fn", [lq.quantizers.ste_sign, lq.quantizers.ste_tern, lq.quantizers.ste_heaviside]
)
@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_ste_grad(fn):
    @np.vectorize
    def ste_grad(x):
        if np.abs(x) <= 1:
            return 1.0
        return 0.0

    x = np.random.uniform(-2, 2, (8, 3, 3, 16))
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        activation = fn(tf_x)
    grad = tape.gradient(activation, tf_x)
    np.testing.assert_allclose(grad.numpy(), ste_grad(x))


@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_swish_grad():
    beta = 10.0

    def swish_grad(x):
        return beta * (2 - beta * x * np.tanh(beta * x / 2)) / (1 + np.cosh(beta * x))

    x = np.random.uniform(-3, 3, (8, 3, 3, 16))
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        activation = lq.quantizers.swish_sign(tf_x, beta=beta)
    grad = tape.gradient(activation, tf_x)
    np.testing.assert_allclose(grad.numpy(), swish_grad(x))


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


@pytest.mark.parametrize(
    "fn", [lq.quantizers.dorefa_quantizer, lq.quantizers.DoReFaQuantizer(2)]
)
def test_dorefa_quantize(fn):
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [fn(x)])
    real_values = np.random.uniform(-2, 2, (2, 5))
    result = f([real_values])[0]
    k_bit = 2
    n = 2 ** k_bit - 1
    assert not np.any(result > 1)
    assert not np.any(result < 0)
    for i in range(n + 1):
        assert np.all(
            result[(result > (2 * i - 1) / (2 * n)) & (result < (2 * i + 1) / (2 * n))]
            == i / n
        )


@pytest.mark.skipif(not tf.executing_eagerly(), reason="requires eager execution")
def test_ste_grad_dorefa():
    @np.vectorize
    def ste_grad(x):
        if x <= 1 and x >= 0:
            return 1.0
        return 0.0

    x = np.random.uniform(-2, 2, (8, 3, 3, 16))
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        activation = lq.quantizers.DoReFaQuantizer(2)(tf_x)
    grad = tape.gradient(activation, tf_x)
    np.testing.assert_allclose(grad.numpy(), ste_grad(x))
