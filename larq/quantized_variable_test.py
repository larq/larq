import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.distribute.values import DistributedVariable

from larq import quantized_scope
from larq.quantized_variable import create_quantized_variable
from larq.testing_utils import evaluate


def get_var(val, dtype=None, name=None):
    return tf.compat.v1.Variable(val, use_resource=True, dtype=dtype, name=name)


@pytest.fixture(params=[True, False])
def quantized(request):
    """pytest fixture for running test quantized and non-quantized"""
    with quantized_scope.scope(request.param):
        yield request.param


def test_quantize_scope(quantized):
    assert quantized_scope.should_quantize() == quantized


def test_inheritance(distribute_scope):
    variable = get_var(3.0)
    quantized_variable = create_quantized_variable(variable)
    assert isinstance(quantized_variable, tf.Variable)
    assert isinstance(quantized_variable, DistributedVariable) is distribute_scope  # type: ignore


def test_overloads(quantized, distribute_scope, eager_and_graph_mode):
    if quantized:
        x = create_quantized_variable(get_var(3.5), quantizer=lambda x: 2 * x)
    else:
        x = create_quantized_variable(get_var(7.0))
    evaluate(x.initializer)
    np.testing.assert_almost_equal(8, evaluate(x + 1))
    np.testing.assert_almost_equal(10, evaluate(3 + x))
    np.testing.assert_almost_equal(14, evaluate(x + x))
    np.testing.assert_almost_equal(5, evaluate(x - 2))
    np.testing.assert_almost_equal(6, evaluate(13 - x))
    np.testing.assert_almost_equal(0, evaluate(x - x))
    np.testing.assert_almost_equal(14, evaluate(x * 2))
    np.testing.assert_almost_equal(21, evaluate(3 * x))
    np.testing.assert_almost_equal(49, evaluate(x * x))
    np.testing.assert_almost_equal(3.5, evaluate(x / 2))
    np.testing.assert_almost_equal(1.5, evaluate(10.5 / x))
    np.testing.assert_almost_equal(3, evaluate(x // 2))
    np.testing.assert_almost_equal(2, evaluate(15 // x))
    np.testing.assert_almost_equal(1, evaluate(x % 2))
    np.testing.assert_almost_equal(2, evaluate(16 % x))
    assert evaluate(x < 12)
    assert evaluate(x <= 12)
    assert not evaluate(x > 12)
    assert not evaluate(x >= 12)
    assert not evaluate(12 < x)
    assert not evaluate(12 <= x)
    assert evaluate(12 > x)
    assert evaluate(12 >= x)
    np.testing.assert_almost_equal(343, evaluate(pow(x, 3)))
    np.testing.assert_almost_equal(128, evaluate(pow(2, x)))
    np.testing.assert_almost_equal(-7, evaluate(-x))
    np.testing.assert_almost_equal(7, evaluate(abs(x)))


def test_tensor_equality(quantized, eager_mode):
    if quantized:
        x = create_quantized_variable(
            get_var([3.5, 4.0, 4.5]), quantizer=lambda x: 2 * x
        )
    else:
        x = create_quantized_variable(get_var([7.0, 8.0, 9.0]))
    evaluate(x.initializer)
    np.testing.assert_array_equal(evaluate(x), [7.0, 8.0, 9.0])
    if int(tf.__version__[0]) >= 2:
        np.testing.assert_array_equal(x == [7.0, 8.0, 10.0], [True, True, False])
        np.testing.assert_array_equal(x != [7.0, 8.0, 10.0], [False, False, True])
