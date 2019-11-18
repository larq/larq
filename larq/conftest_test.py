import tensorflow as tf


def test_eager_and_graph_mode_fixture(eager_and_graph_mode):
    assert tf.executing_eagerly() == (eager_and_graph_mode == "eager")


def test_eager_mode_fixture(eager_mode):
    assert tf.executing_eagerly()


def test_graph_mode_fixture(graph_mode):
    assert not tf.executing_eagerly()


def test_distribute_scope(distribute_scope):
    assert tf.distribute.has_strategy() is distribute_scope
