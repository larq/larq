import numpy as np
import pytest
import tensorflow as tf
from packaging import version

import larq as lq
from larq import testing_utils


class DummyTrainableQuantizer(tf.keras.layers.Layer):
    """Used to test whether we can set layers as quantizers without any throws."""

    _custom_metrics = None

    def build(self, input_shape):
        self.dummy_weight = self.add_weight("dummy_weight", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return self.dummy_weight * inputs


class TestCommonFunctionality:
    """Test functionality common to all quantizers, like serialization and usage."""

    @pytest.mark.parametrize("module", [lq.quantizers, tf.keras.activations])
    @pytest.mark.parametrize(
        "name,ref_cls",
        [
            ("ste_sign", lq.quantizers.SteSign),
            ("approx_sign", lq.quantizers.ApproxSign),
            ("ste_heaviside", lq.quantizers.SteHeaviside),
            ("magnitude_aware_sign", lq.quantizers.MagnitudeAwareSign),
            ("swish_sign", lq.quantizers.SwishSign),
            ("ste_tern", lq.quantizers.SteTern),
        ],
    )
    def test_serialization(self, module, name, ref_cls):
        fn = module.get(name)
        assert fn.__class__ == ref_cls
        fn = module.get(ref_cls())
        assert fn.__class__ == ref_cls
        assert type(fn.precision) == int
        if module == tf.keras.activations and version.parse(
            tf.__version__
        ) < version.parse("1.15"):
            pytest.skip(
                "TensorFlow < 1.15 does not support Quantizer classes as activations"
            )
        config = module.serialize(fn)
        fn = module.deserialize(config)
        assert fn.__class__ == ref_cls
        assert type(fn.precision) == int

    def test_noop_serialization(self):
        fn = lq.quantizers.get(lq.quantizers.NoOp(precision=1))
        assert fn.__class__ == lq.quantizers.NoOp
        assert fn.precision == 1
        config = lq.quantizers.serialize(fn)
        fn = lq.quantizers.deserialize(config)
        assert fn.__class__ == lq.quantizers.NoOp
        assert fn.precision == 1

    def test_invalid_usage(self):
        with pytest.raises(ValueError):
            lq.quantizers.get(42)
        with pytest.raises(ValueError):
            lq.quantizers.get("unknown")

    @pytest.mark.parametrize("quantizer", ["input_quantizer", "kernel_quantizer"])
    def test_layer_as_quantizer(self, quantizer, keras_should_run_eagerly):
        """Test whether a keras.layers.Layer can be used as quantizer."""

        input_data = testing_utils.random_input((1, 10))

        model = tf.keras.Sequential(
            [lq.layers.QuantDense(1, **{quantizer: DummyTrainableQuantizer()})]
        )
        model.compile(optimizer="sgd", loss="mse", run_eagerly=keras_should_run_eagerly)
        model.fit(input_data, np.ones((1,)), epochs=1)

        assert any(["dummy_weight" in var.name for var in model.trainable_variables])


class TestQuantization:
    """Test binarization and ternarization."""

    @pytest.mark.parametrize(
        "fn",
        [
            "ste_sign",
            lq.quantizers.SteSign(),
            "approx_sign",
            lq.quantizers.ApproxSign(),
            "swish_sign",
            lq.quantizers.SwishSign(),
        ],
    )
    def test_xnor_binarization(self, fn):
        x = tf.keras.backend.placeholder(ndim=2)
        f = tf.keras.backend.function([x], [lq.quantizers.get(fn)(x)])
        binarized_values = np.random.choice([-1, 1], size=(2, 5))
        result = f([binarized_values])[0]
        np.testing.assert_allclose(result, binarized_values)

        real_values = testing_utils.generate_real_values_with_zeros()
        result = f([real_values])[0]
        assert not np.any(result == 0)
        assert np.all(result[real_values < 0] == -1)
        assert np.all(result[real_values >= 0] == 1)

        zero_values = np.zeros((2, 5))
        result = f([zero_values])[0]
        assert np.all(result == 1)

    @pytest.mark.parametrize("fn", ["ste_heaviside", lq.quantizers.SteHeaviside()])
    def test_and_binarization(self, fn):
        x = tf.keras.backend.placeholder(ndim=2)
        f = tf.keras.backend.function([x], [lq.quantizers.get(fn)(x)])

        binarized_values = np.random.choice([0, 1], size=(2, 5))
        result = f([binarized_values])[0]
        np.testing.assert_allclose(result, binarized_values)

        real_values = testing_utils.generate_real_values_with_zeros()
        result = f([real_values])[0]
        assert np.all(result[real_values <= 0] == 0)
        assert np.all(result[real_values > 0] == 1)

    @pytest.mark.usefixtures("eager_mode")
    def test_magnitude_aware_sign_binarization(self):
        a = np.random.uniform(-2, 2, (3, 2, 2, 3))
        x = tf.Variable(a)
        y = lq.quantizers.MagnitudeAwareSign()(x)

        assert y.shape == x.shape

        # check sign
        np.testing.assert_allclose(tf.sign(y).numpy(), np.sign(a))

        # check magnitude
        np.testing.assert_allclose(
            tf.reduce_mean(tf.abs(y), axis=[0, 1, 2]).numpy(),
            [np.mean(np.reshape(np.abs(a[:, :, :, i]), [-1])) for i in range(3)],
        )

    @pytest.mark.parametrize(
        "fn",
        [
            "ste_tern",
            lq.quantizers.SteTern(),
            lq.quantizers.SteTern(ternary_weight_networks=True),
            lq.quantizers.SteTern(threshold_value=np.random.uniform(0.01, 0.8)),
        ],
    )
    def test_ternarization_basic(self, fn):
        x = tf.keras.backend.placeholder(ndim=2)
        f = tf.keras.backend.function([x], [lq.quantizers.get(fn)(x)])

        ternarized_values = np.random.choice([-1, 0, 1], size=(4, 10))
        result = f([ternarized_values])[0]
        np.testing.assert_allclose(result, ternarized_values)
        assert not np.any(result > 1)
        assert not np.any(result < -1)
        assert np.any(result == -1)
        assert np.any(result == 1)
        assert np.any(result == 0)

        real_values = testing_utils.generate_real_values_with_zeros()
        result = f([real_values])[0]
        assert not np.any(result > 1)
        assert not np.any(result < -1)
        assert np.any(result == -1)
        assert np.any(result == 1)
        assert np.any(result == 0)

    @pytest.mark.parametrize("fn", ["ste_tern", lq.quantizers.SteTern()])
    def test_ternarization_with_default_threshold(self, fn):
        x = tf.keras.backend.placeholder(ndim=2)
        test_threshold = 0.05  # This is the default
        f = tf.keras.backend.function([x], [lq.quantizers.get(fn)(x)])

        real_values = testing_utils.generate_real_values_with_zeros()
        result = f([real_values])[0]
        assert np.all(result[real_values > test_threshold] == 1)
        assert np.all(result[real_values < -test_threshold] == -1)
        assert np.all(result[np.abs(real_values) < test_threshold] == 0)
        assert not np.any(result > 1)
        assert not np.any(result < -1)

    def test_ternarization_with_custom_threshold(self):
        x = tf.keras.backend.placeholder(ndim=2)
        test_threshold = np.random.uniform(0.01, 0.8)
        fn = lq.quantizers.SteTern(threshold_value=test_threshold)
        f = tf.keras.backend.function([x], [fn(x)])

        real_values = testing_utils.generate_real_values_with_zeros()
        result = f([real_values])[0]
        assert np.all(result[real_values > test_threshold] == 1)
        assert np.all(result[real_values < -test_threshold] == -1)
        assert np.all(result[np.abs(real_values) < test_threshold] == 0)
        assert not np.any(result > 1)
        assert not np.any(result < -1)

    def test_ternarization_with_ternary_weight_networks(self):
        x = tf.keras.backend.placeholder(ndim=2)
        real_values = testing_utils.generate_real_values_with_zeros()
        test_threshold = 0.7 * np.sum(np.abs(real_values)) / np.size(real_values)
        fn = lq.quantizers.SteTern(ternary_weight_networks=True)
        f = tf.keras.backend.function([x], [fn(x)])

        result = f([real_values])[0]
        assert np.all(result[real_values > test_threshold] == 1)
        assert np.all(result[real_values < -test_threshold] == -1)
        assert np.all(result[np.abs(real_values) < test_threshold] == 0)
        assert not np.any(result > 1)
        assert not np.any(result < -1)

    def test_dorefa_quantize(self):
        x = tf.keras.backend.placeholder(ndim=2)
        f = tf.keras.backend.function([x], [lq.quantizers.DoReFa(2)(x)])
        real_values = testing_utils.generate_real_values_with_zeros()
        result = f([real_values])[0]
        k_bit = 2
        n = 2 ** k_bit - 1
        assert not np.any(result > 1)
        assert not np.any(result < 0)
        for i in range(n + 1):
            assert np.all(
                result[
                    (real_values > (2 * i - 1) / (2 * n))
                    & (real_values < (2 * i + 1) / (2 * n))
                ]
                == i / n
            )


@pytest.mark.usefixtures("eager_mode")
class TestGradients:
    """Test gradients for different quantizers."""

    @pytest.mark.parametrize(
        "fn",
        [
            lq.quantizers.SteSign(clip_value=None),
            lq.quantizers.SteTern(clip_value=None),
            lq.quantizers.SteHeaviside(clip_value=None),
        ],
    )
    def test_identity_ste_grad(self, fn):
        x = testing_utils.generate_real_values_with_zeros(shape=(8, 3, 3, 16))
        tf_x = tf.Variable(x)
        with tf.GradientTape() as tape:
            activation = fn(tf_x)
        grad = tape.gradient(activation, tf_x)
        np.testing.assert_allclose(grad.numpy(), np.ones_like(x))

    @pytest.mark.parametrize(
        "fn",
        [
            lq.quantizers.SteSign(),
            lq.quantizers.SteTern(),
            lq.quantizers.SteHeaviside(),
        ],
    )
    def test_ste_grad(self, fn):
        @np.vectorize
        def ste_grad(x):
            if np.abs(x) <= 1:
                return 1.0
            return 0.0

        x = testing_utils.generate_real_values_with_zeros(shape=(8, 3, 3, 16))
        tf_x = tf.Variable(x)
        with tf.GradientTape() as tape:
            activation = fn(tf_x)
        grad = tape.gradient(activation, tf_x)
        np.testing.assert_allclose(grad.numpy(), ste_grad(x))

    # Test with and without default threshold
    def test_swish_grad(self):
        def swish_grad(x, beta):
            return (
                beta * (2 - beta * x * np.tanh(beta * x / 2)) / (1 + np.cosh(beta * x))
            )

        x = testing_utils.generate_real_values_with_zeros(shape=(8, 3, 3, 16))
        tf_x = tf.Variable(x)
        with tf.GradientTape() as tape:
            activation = lq.quantizers.SwishSign()(tf_x)
        grad = tape.gradient(activation, tf_x)
        np.testing.assert_allclose(grad.numpy(), swish_grad(x, beta=5.0))

        with tf.GradientTape() as tape:
            activation = lq.quantizers.SwishSign(beta=10.0)(tf_x)
        grad = tape.gradient(activation, tf_x)
        np.testing.assert_allclose(grad.numpy(), swish_grad(x, beta=10.0))

    def test_approx_sign_grad(self):
        @np.vectorize
        def approx_sign_grad(x):
            if np.abs(x) <= 1:
                return 2 - 2 * np.abs(x)
            return 0.0

        x = testing_utils.generate_real_values_with_zeros(shape=(8, 3, 3, 16))
        tf_x = tf.Variable(x)
        with tf.GradientTape() as tape:
            activation = lq.quantizers.ApproxSign()(tf_x)
        grad = tape.gradient(activation, tf_x)
        np.testing.assert_allclose(grad.numpy(), approx_sign_grad(x))

    def test_magnitude_aware_sign_grad(self):
        a = np.random.uniform(-2, 2, (3, 2, 2, 3))
        x = tf.Variable(a)
        with tf.GradientTape() as tape:
            y = lq.quantizers.MagnitudeAwareSign()(x)
        grad = tape.gradient(y, x)

        scale_vector = [
            np.mean(np.reshape(np.abs(a[:, :, :, i]), [-1])) for i in range(3)
        ]

        np.testing.assert_allclose(
            grad.numpy(), np.where(abs(a) < 1, np.ones(a.shape) * scale_vector, 0)
        )

    def test_dorefa_ste_grad(self):
        @np.vectorize
        def ste_grad(x):
            if x <= 1 and x >= 0:
                return 1.0
            return 0.0

        x = testing_utils.generate_real_values_with_zeros(shape=(8, 3, 3, 16))
        tf_x = tf.Variable(x)
        with tf.GradientTape() as tape:
            activation = lq.quantizers.DoReFa(2)(tf_x)
        grad = tape.gradient(activation, tf_x)
        np.testing.assert_allclose(grad.numpy(), ste_grad(x))


@pytest.mark.parametrize(
    "quantizer",
    [
        ("ste_sign", lq.quantizers.SteSign),
        ("approx_sign", lq.quantizers.ApproxSign),
        ("ste_heaviside", lq.quantizers.SteHeaviside),
        ("swish_sign", lq.quantizers.SwishSign),
        ("magnitude_aware_sign", lq.quantizers.MagnitudeAwareSign),
        ("ste_tern", lq.quantizers.SteTern),
        ("dorefa_quantizer", lq.quantizers.DoReFa),
    ],
)
def test_metrics(quantizer):
    quantizer_str, quantizer_cls = quantizer

    # No metric
    model = tf.keras.models.Sequential(
        [lq.layers.QuantDense(3, kernel_quantizer=quantizer_str, input_shape=(32,))]
    )
    model.compile(loss="mse", optimizer="sgd")
    assert len(model.layers[0]._metrics) == 0

    # Metric added using scope
    with lq.context.metrics_scope(["flip_ratio"]):
        model = tf.keras.models.Sequential(
            [lq.layers.QuantDense(3, kernel_quantizer=quantizer_str, input_shape=(32,))]
        )
    model.compile(loss="mse", optimizer="sgd")

    if version.parse(tf.__version__) > version.parse("1.14"):
        assert len(model.layers[0].kernel_quantizer._metrics) == 1
    else:
        # In TF1.14, call() gets called twice, resulting in having an extra initial
        # metrics copy.
        assert len(model.layers[0].kernel_quantizer._metrics) == 2

    # Metric added explicitly to quantizer
    model = tf.keras.models.Sequential(
        [
            lq.layers.QuantDense(
                3,
                kernel_quantizer=quantizer_cls(metrics=["flip_ratio"]),
                input_shape=(32,),
            )
        ]
    )
    model.compile(loss="mse", optimizer="sgd")
    if version.parse(tf.__version__) > version.parse("1.14"):
        assert len(model.layers[0].kernel_quantizer._metrics) == 1
    else:
        # In TF1.14, call() gets called twice, resulting in having an extra initial
        # metrics copy.
        assert len(model.layers[0].kernel_quantizer._metrics) == 2


def test_get_kernel_quantizer_assigns_metrics():
    with lq.context.metrics_scope(["flip_ratio"]):
        ste_sign = lq.quantizers.get_kernel_quantizer("ste_sign")
        assert "flip_ratio" in lq.context.get_training_metrics()

    assert isinstance(ste_sign, lq.quantizers.SteSign)
    assert "flip_ratio" in ste_sign._custom_metrics


def test_get_kernel_quantizer_accepts_function():
    custom_quantizer = lq.quantizers.get_kernel_quantizer(lambda x: x)
    assert callable(custom_quantizer)
    assert not hasattr(custom_quantizer, "_custom_metrics")


def test_backwards_compat_aliases():
    assert lq.quantizers.DoReFaQuantizer == lq.quantizers.DoReFa
    assert lq.quantizers.NoOpQuantizer == lq.quantizers.NoOp
