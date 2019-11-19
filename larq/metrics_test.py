import pytest
import tensorflow as tf

from larq import metrics


def test_scope():
    assert metrics.get_training_metrics() == set()
    with metrics.scope(["flip_ratio"]):
        assert metrics.get_training_metrics() == {"flip_ratio"}
    assert metrics.get_training_metrics() == set()
    with pytest.raises(ValueError, match=r".*unknown_metric.*"):
        with metrics.scope(["flip_ratio", "unknown_metric"]):
            pass


def test_config():
    mcv = metrics.FlipRatio(
        values_shape=[3, 3], values_dtype="int16", name="mcv", dtype=tf.float16
    )
    assert mcv.name == "mcv"
    assert mcv.stateful
    assert mcv.dtype == tf.float16
    assert mcv.values_dtype == tf.int16
    assert mcv.values_shape == [3, 3]

    mcv2 = metrics.FlipRatio.from_config(mcv.get_config())
    assert mcv2.name == "mcv"
    assert mcv2.stateful
    assert mcv2.dtype == tf.float16
    assert mcv2.values_dtype == tf.int16
    assert mcv2.values_shape == [3, 3]


def test_metric(eager_mode):
    mcv = metrics.FlipRatio([2])

    assert 0 == mcv.result().numpy()
    assert 0 == mcv.total.numpy()
    assert 0 == mcv.count.numpy()

    mcv.update_state([1, 1])
    assert all([1, 1] == mcv._previous_values.numpy())
    assert 0 == mcv.total.numpy()
    assert 1 == mcv.count.numpy()
    assert 0 == mcv.result().numpy()

    mcv.update_state([2, 2])
    assert all([2, 2] == mcv._previous_values.numpy())
    assert 1 == mcv.total.numpy()
    assert 2 == mcv.count.numpy()
    assert 1 == mcv.result().numpy()

    mcv.update_state([1, 2])
    assert all([1, 2] == mcv._previous_values.numpy())
    assert 1.5 == mcv.total.numpy()
    assert 3 == mcv.count.numpy()
    assert 1.5 / 2 == mcv.result().numpy()


def test_metric_in_graph_mode(graph_mode):
    mcv = metrics.FlipRatio([2])

    new_state = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2])
    update_state_op = mcv.update_state(new_state)
    metric_value = mcv.result()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.variables_initializer(mcv.variables))

        sess.run(update_state_op, feed_dict={new_state: [1, 1]})
        sess.run(update_state_op, feed_dict={new_state: [2, 2]})
        sess.run(update_state_op, feed_dict={new_state: [1, 2]})

        previous, total, count, result = sess.run(
            [mcv._previous_values, mcv.total, mcv.count, metric_value]
        )

    assert all([1, 2] == previous)
    assert 1.5 == total
    assert 3 == count
    assert 1.5 / 2 == result
