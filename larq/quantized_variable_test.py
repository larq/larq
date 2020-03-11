import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal, assert_array_equal
from packaging import version
from tensorflow.python.distribute.values import DistributedVariable

from larq import context
from larq.quantized_variable import QuantizedVariable
from larq.testing_utils import evaluate


def get_var(val, dtype=None, name=None):
    return tf.compat.v1.Variable(val, use_resource=True, dtype=dtype, name=name)


def test_inheritance(distribute_scope):
    variable = get_var(3.0)
    quantized_variable = QuantizedVariable.from_variable(variable)
    assert isinstance(quantized_variable, QuantizedVariable)
    assert isinstance(quantized_variable, tf.Variable)
    assert isinstance(quantized_variable, DistributedVariable) is distribute_scope  # type: ignore


def test_read(eager_and_graph_mode, distribute_scope):
    x = QuantizedVariable.from_variable(get_var(3.5), quantizer=lambda x: 2 * x)
    evaluate(x.initializer)

    assert evaluate(x) == 3.5
    assert evaluate(x.value()) == 3.5
    assert evaluate(x.read_value()) == 3.5
    assert evaluate(tf.identity(x)) == 3.5

    with context.quantized_scope(True):
        assert evaluate(x) == 7
        assert evaluate(x.value()) == 7
        assert evaluate(x.read_value()) == 7
        assert evaluate(tf.identity(x)) == 7


def test_sparse_reads(eager_and_graph_mode):
    x = QuantizedVariable.from_variable(get_var([1.0, 2.0]), quantizer=lambda x: 2 * x)
    evaluate(x.initializer)

    assert evaluate(x.sparse_read([0])) == 1
    assert evaluate(x.gather_nd([0])) == 1
    with context.quantized_scope(True):
        assert evaluate(x.sparse_read([0])) == 2
        assert evaluate(x.gather_nd([0])) == 2


def test_read_nested_scopes(eager_and_graph_mode, distribute_scope):
    x = QuantizedVariable.from_variable(get_var(3.5), quantizer=lambda x: 2 * x)
    evaluate(x.initializer)
    with context.quantized_scope(True):
        assert evaluate(x.read_value()) == 7
        with context.quantized_scope(False):
            assert evaluate(x.read_value()) == 3.5
        assert evaluate(x.read_value()) == 7


def test_method_delegations(eager_and_graph_mode, distribute_scope):
    x = QuantizedVariable.from_variable(get_var(3.5), quantizer=lambda x: 2 * x)
    with context.quantized_scope(True):
        evaluate(x.initializer)
        assert evaluate(x.value()) == 7
        assert evaluate(x.read_value()) == 7
        assert x.trainable
        if version.parse(tf.__version__) > version.parse("1.14"):
            assert x.synchronization == x.latent_variable.synchronization
        assert x.aggregation == x.latent_variable.aggregation
        assert evaluate(x.initialized_value()) == 7
        if not tf.executing_eagerly():
            if not distribute_scope:
                # These functions are not supported for DistributedVariables
                x.load(4.5)
                assert x.eval() == 9
            assert evaluate(x.initial_value) == 7
            assert x.op == x.latent_variable.op
            assert x.graph == x.latent_variable.graph
        if not distribute_scope:
            # These attributes are not supported for DistributedVariables
            assert x.constraint is None
            assert x.initializer == x.latent_variable.initializer
        assert evaluate(x.assign(4)) == 8
        assert evaluate(x.assign_add(1)) == 10
        assert evaluate(x.assign_sub(1.5)) == 7
        assert x.name == x.latent_variable.name
        assert x.device == x.latent_variable.device
        assert x.shape == ()
        assert x.get_shape() == ()


def test_scatter_method_delegations(eager_and_graph_mode):
    x = QuantizedVariable.from_variable(get_var([3.5, 4]), quantizer=lambda x: 2 * x)
    evaluate(x.initializer)
    with context.quantized_scope(True):
        assert_array_equal(evaluate(x.value()), [7, 8])

        def slices(val, index):
            return tf.IndexedSlices(
                values=tf.constant(val, dtype=tf.float32),
                indices=tf.constant(index, dtype=tf.int32),
                dense_shape=tf.constant([2], dtype=tf.int32),
            )

        assert_array_equal(evaluate(x.scatter_sub(slices(0.5, 0))), [6, 8])
        assert_array_equal(evaluate(x.scatter_add(slices(0.5, 0))), [7, 8])
        if version.parse(tf.__version__) > version.parse("1.14"):
            assert_array_equal(evaluate(x.scatter_max(slices(4.5, 1))), [7, 9])
            assert_array_equal(evaluate(x.scatter_min(slices(4.0, 1))), [7, 8])
            assert_array_equal(evaluate(x.scatter_mul(slices(2.0, 1))), [7, 16])
            assert_array_equal(evaluate(x.scatter_div(slices(2.0, 1))), [7, 8])
        assert_array_equal(evaluate(x.scatter_update(slices(2, 1))), [7, 4])
        assert_array_equal(evaluate(x.scatter_nd_sub([[0], [1]], [0.5, 1.0])), [6, 2])
        assert_array_equal(evaluate(x.scatter_nd_add([[0], [1]], [0.5, 1.0])), [7, 4])
        assert_array_equal(
            evaluate(x.scatter_nd_update([[0], [1]], [0.5, 1.0])), [1, 2]
        )


def test_overloads(eager_and_graph_mode, quantized, distribute_scope):
    if quantized:
        x = QuantizedVariable.from_variable(get_var(3.5), quantizer=lambda x: 2 * x)
    else:
        x = QuantizedVariable.from_variable(get_var(7.0))
    evaluate(x.initializer)
    assert_almost_equal(8, evaluate(x + 1))
    assert_almost_equal(10, evaluate(3 + x))
    assert_almost_equal(14, evaluate(x + x))
    assert_almost_equal(5, evaluate(x - 2))
    assert_almost_equal(6, evaluate(13 - x))
    assert_almost_equal(0, evaluate(x - x))
    assert_almost_equal(14, evaluate(x * 2))
    assert_almost_equal(21, evaluate(3 * x))
    assert_almost_equal(49, evaluate(x * x))
    assert_almost_equal(3.5, evaluate(x / 2))
    assert_almost_equal(1.5, evaluate(10.5 / x))
    assert_almost_equal(3, evaluate(x // 2))
    assert_almost_equal(2, evaluate(15 // x))
    assert_almost_equal(1, evaluate(x % 2))
    assert_almost_equal(2, evaluate(16 % x))
    assert evaluate(x < 12)
    assert evaluate(x <= 12)
    assert not evaluate(x > 12)
    assert not evaluate(x >= 12)
    assert not evaluate(12 < x)
    assert not evaluate(12 <= x)
    assert evaluate(12 > x)
    assert evaluate(12 >= x)
    assert_almost_equal(343, evaluate(pow(x, 3)))
    assert_almost_equal(128, evaluate(pow(2, x)))
    assert_almost_equal(-7, evaluate(-x))
    assert_almost_equal(7, evaluate(abs(x)))


def test_tensor_equality(quantized, eager_mode):
    if quantized:
        x = QuantizedVariable.from_variable(
            get_var([3.5, 4.0, 4.5]), quantizer=lambda x: 2 * x
        )
    else:
        x = QuantizedVariable.from_variable(get_var([7.0, 8.0, 9.0]))
    evaluate(x.initializer)
    assert_array_equal(evaluate(x), [7.0, 8.0, 9.0])
    if version.parse(tf.__version__) >= version.parse("2"):
        assert_array_equal(x == [7.0, 8.0, 10.0], [True, True, False])
        assert_array_equal(x != [7.0, 8.0, 10.0], [False, False, True])


def test_assign(eager_and_graph_mode, quantized, distribute_scope):
    x = QuantizedVariable.from_variable(
        get_var(0.0, tf.float64), quantizer=lambda x: 2 * x
    )
    evaluate(x.initializer)

    latent_value = 3.14
    value = latent_value * 2 if quantized else latent_value

    # Assign float32 values
    lv = tf.constant(latent_value, dtype=tf.float64)
    assert_almost_equal(evaluate(x.assign(lv)), value)
    assert_almost_equal(evaluate(x.assign_add(lv)), value * 2)
    assert_almost_equal(evaluate(x.assign_sub(lv)), value)

    # Assign Python floats
    assert_almost_equal(evaluate(x.assign(0.0)), 0.0)
    assert_almost_equal(evaluate(x.assign(latent_value)), value)
    assert_almost_equal(evaluate(x.assign_add(latent_value)), value * 2)
    assert_almost_equal(evaluate(x.assign_sub(latent_value)), value)

    # Assign multiple times
    assign = x.assign(0.0)
    assert_almost_equal(evaluate(assign), 0.0)
    assert_almost_equal(evaluate(assign.assign(latent_value)), value)
    if version.parse(tf.__version__) >= version.parse("2.2"):
        assert_almost_equal(
            evaluate(x.assign_add(latent_value).assign_add(latent_value)), value * 3
        )
        assert_almost_equal(evaluate(x), value * 3)
        assert_almost_equal(
            evaluate(x.assign_sub(latent_value).assign_sub(latent_value)), value
        )
        assert_almost_equal(evaluate(x), value)

    # Assign with read_value=False
    assert_almost_equal(evaluate(x.assign(0.0)), 0.0)
    assert evaluate(x.assign(latent_value, read_value=False)) is None
    assert_almost_equal(evaluate(x), value)
    assert evaluate(x.assign_add(latent_value, read_value=False)) is None
    assert_almost_equal(evaluate(x), 2 * value)
    assert evaluate(x.assign_sub(latent_value, read_value=False)) is None
    assert_almost_equal(evaluate(x), value)

    # Use the tf.assign functions instead of the var.assign methods.
    assert_almost_equal(evaluate(tf.compat.v1.assign(x, 0.0)), 0.0)
    assert_almost_equal(evaluate(tf.compat.v1.assign(x, latent_value)), value)
    assert_almost_equal(evaluate(tf.compat.v1.assign_add(x, latent_value)), value * 2)
    assert_almost_equal(evaluate(tf.compat.v1.assign_sub(x, latent_value)), value)


def test_checkpoint(tmp_path, eager_and_graph_mode):
    x = QuantizedVariable.from_variable(get_var(0.0), quantizer=lambda x: 2 * x)
    evaluate(x.initializer)
    evaluate(x.assign(123.0))

    checkpoint = tf.train.Checkpoint(x=x)
    save_path = checkpoint.save(tmp_path)
    evaluate(x.assign(234.0))
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    assert isinstance(x, QuantizedVariable)
    assert evaluate(x) == 123.0
    with context.quantized_scope(True):
        assert evaluate(x) == 123.0 * 2


def test_invalid_wrapped_usage(distribute_scope):
    with pytest.raises(ValueError, match="`variable` must be of type"):
        QuantizedVariable.from_variable(tf.constant([1.0]))
    with pytest.raises(ValueError, match="`quantizer` must be `callable` or `None`"):
        QuantizedVariable.from_variable(get_var([1.0]), 1)  # type: ignore
    with pytest.raises(ValueError, match="`precision` must be of type `int` or `None`"):
        QuantizedVariable.from_variable(get_var([1.0]), precision=1.0)  # type: ignore


def test_repr(snapshot, eager_and_graph_mode):
    x = get_var(0.0, name="x")

    class Quantizer:
        def __call__(self, x):
            return x

    snapshot.assert_match(
        repr(QuantizedVariable.from_variable(x, quantizer=lambda x: 2 * x))
    )
    snapshot.assert_match(
        repr(QuantizedVariable.from_variable(x, quantizer=Quantizer()))
    )
    snapshot.assert_match(repr(QuantizedVariable.from_variable(x, precision=1)))


@pytest.mark.parametrize("should_quantize", [True, False])
def test_optimizer(eager_mode, should_quantize):
    x = QuantizedVariable.from_variable(get_var(1.0), quantizer=lambda x: -x)
    opt = tf.keras.optimizers.SGD(1.0)

    def loss():
        with context.quantized_scope(should_quantize):
            return x + 1.0

    @tf.function
    def f():
        opt.minimize(loss, var_list=[x])

    f()
    if should_quantize:
        assert evaluate(x) == 2.0
        with context.quantized_scope(should_quantize):
            assert evaluate(x) == -2.0
    else:
        assert evaluate(x) == 0.0
