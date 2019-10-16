import numpy as np
import larq as lq
from larq.testing_utils import generate_real_values_with_zeros


def test_sign():
    binarized_values = np.random.choice([-1, 1], size=(2, 5)).astype(np.float32)
    result = lq.math.sign(binarized_values)
    np.testing.assert_allclose(result, binarized_values)

    real_values = generate_real_values_with_zeros()
    result = lq.math.sign(real_values)
    assert not np.any(result == 0)
    assert np.all(result[real_values < 0] == -1)
    assert np.all(result[real_values >= 0] == 1)

    zero_values = np.zeros((2, 5))
    result = lq.math.sign(zero_values)
    assert np.all(result == 1)
