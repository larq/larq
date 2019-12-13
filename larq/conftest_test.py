import tensorflow as tf

from larq import quantized_scope


def test_eager_and_graph_mode_fixture(eager_and_graph_mode):
    if eager_and_graph_mode == "eager":
        assert tf.executing_eagerly()
    else:
        assert not tf.executing_eagerly()
        assert tf.compat.v1.get_default_session() is not None


def test_eager_mode_fixture(eager_mode):
    assert tf.executing_eagerly()


def test_graph_mode_fixture(graph_mode):
    assert not tf.executing_eagerly()
    assert tf.compat.v1.get_default_session() is not None


def test_distribute_scope(distribute_scope):
    assert tf.distribute.has_strategy() is distribute_scope


def test_quantize_scope(quantized):
    assert quantized_scope.should_quantize() == quantized
