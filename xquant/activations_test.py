import tensorflow as tf
import numpy as np
import pytest
import xquant as xq


@pytest.mark.parametrize("name", ["hard_tanh"])
def test_serialization(name):
    fn = tf.keras.activations.get(name)
    ref_fn = getattr(xq.activations, name)
    assert fn == ref_fn
    config = tf.keras.activations.serialize(fn)
    fn = tf.keras.activations.deserialize(config)
    assert fn == ref_fn


def test_hard_tanh():
    real_values = np.random.uniform(-2, 2, (3, 3, 32))
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [xq.activations.hard_tanh(x)])
    result = f([real_values])[0]
    np.testing.assert_allclose(result, np.clip(real_values, -1, 1))
