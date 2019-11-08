import tensorflow as tf


def test_eager_and_graph_mode_fixture(eager_and_graph_mode):
    if eager_and_graph_mode == "eager":
        assert tf.executing_eagerly()
    else:
        assert not tf.executing_eagerly()


def test_eager_mode_fixture(eager_mode):
    assert tf.executing_eagerly()


def test_graph_mode_fixture(graph_mode):
    assert not tf.executing_eagerly()
