import numpy as np
import pytest
import tensorflow as tf

import larq as lq
from larq.testing_utils import generate_real_values_with_zeros


@pytest.mark.parametrize("name", ["hard_tanh", "leaky_tanh"])
def test_serialization(name):
    fn = tf.keras.activations.get(name)
    ref_fn = getattr(lq.activations, name)
    assert fn == ref_fn
    config = tf.keras.activations.serialize(fn)
    fn = tf.keras.activations.deserialize(config)
    assert fn == ref_fn


def test_hard_tanh():
    real_values = generate_real_values_with_zeros()
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [lq.activations.hard_tanh(x)])
    result = f([real_values])[0]
    np.testing.assert_allclose(result, np.clip(real_values, -1, 1))


def test_leaky_tanh():
    @np.vectorize
    def leaky_tanh(x, alpha):
        if x <= -1:
            return -1 + alpha * (x + 1)
        elif x <= 1:
            return x
        else:
            return 1 + alpha * (x - 1)

    real_values = generate_real_values_with_zeros()
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [lq.activations.leaky_tanh(x)])
    result = f([real_values])[0]
    np.testing.assert_allclose(result, leaky_tanh(real_values, alpha=0.2))
