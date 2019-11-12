import pytest
import tensorflow as tf
from tensorflow.python.eager import context


@pytest.fixture
def eager_mode():
    """pytest fixture for running test in eager mode"""
    with context.eager_mode():
        yield


@pytest.fixture
def graph_mode():
    """pytest fixture for running test in graph mode"""
    with context.graph_mode():
        yield


@pytest.fixture(params=["eager", "graph"])
def eager_and_graph_mode(request):
    """pytest fixture for running test in eager and graph mode"""
    with getattr(context, f"{request.param}_mode")():
        yield request.param


@pytest.fixture(params=["graph", "tf_eager", "tf_keras_eager"])
def keras_should_run_eagerly(request):
    """Fixture to run in graph and two eager modes.

    The modes are:
    - Graph mode
    - TensorFlow eager and Keras eager
    - TensorFlow eager and Keras not eager

    The `tf.context` sets graph/eager mode for TensorFlow. The yield is True if Keras
    should run eagerly.
    """

    if request.param == "graph":
        if int(tf.__version__[0]) >= 2:
            pytest.skip("Skipping graph mode for TensorFlow 2+.")

        with context.graph_mode():
            yield
    else:
        with context.eager_mode():
            yield request.param == "tf_keras_eager"
