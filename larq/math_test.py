import numpy as np
import pytest
import tensorflow as tf

import larq as lq
from larq.testing_utils import generate_real_values_with_zeros


@pytest.mark.parametrize("fn", [lq.math.sign])
def test_sign(fn):
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [fn(x)])
    binarized_values = np.random.choice([-1, 1], size=(2, 5)).astype(np.float32)
    result = f(binarized_values)[0]
    np.testing.assert_allclose(result, binarized_values)

    real_values = generate_real_values_with_zeros()
    result = f(real_values)[0]
    assert not np.any(result == 0)
    assert np.all(result[real_values < 0] == -1)
    assert np.all(result[real_values >= 0] == 1)

    zero_values = np.zeros((2, 5))
    result = f(zero_values)[0]
    assert np.all(result == 1)


@pytest.mark.parametrize("fn", [lq.math.heaviside])
def test_heaviside(fn):
    x = tf.keras.backend.placeholder(ndim=2)
    f = tf.keras.backend.function([x], [fn(x)])
    binarized_values = np.random.choice([0, 1], size=(2, 5))
    result = f([binarized_values])[0]
    np.testing.assert_allclose(result, binarized_values)

    real_values = generate_real_values_with_zeros()
    result = f([real_values])[0]
    assert np.all(result[real_values <= 0] == 0)
    assert np.all(result[real_values > 0] == 1)
