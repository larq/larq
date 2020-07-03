import pytest
import tensorflow as tf
from packaging import version
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import context

from larq import context as lq_context

if version.parse(tf.__version__) >= version.parse("1.15"):
    strategy_combinations.set_virtual_cpus_to_at_least(3)
    distributed_devices = ["/cpu:1", "/cpu:2"]
else:
    distributed_devices = ["/cpu:0"]


@pytest.fixture
def eager_mode():
    """pytest fixture for running test in eager mode"""
    with context.eager_mode():
        yield


@pytest.fixture
def graph_mode():
    """pytest fixture for running test in graph mode"""
    with context.graph_mode():
        with tf.compat.v1.Session().as_default():
            yield
            tf.keras.backend.clear_session()


@pytest.fixture(params=["eager", "graph"])
def eager_and_graph_mode(request):
    """pytest fixture for running test in eager and graph mode"""
    if request.param == "graph":
        with context.graph_mode():
            with tf.compat.v1.Session().as_default():
                yield request.param
                tf.keras.backend.clear_session()
    else:
        with context.eager_mode():
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
        if version.parse(tf.__version__) >= version.parse("2"):
            pytest.skip("Skipping graph mode for TensorFlow 2+.")

        with context.graph_mode():
            yield
    else:
        with context.eager_mode():
            yield request.param == "tf_keras_eager"


@pytest.fixture(params=[False, True])
def distribute_scope(request):
    if request.param is True:
        with tf.distribute.MirroredStrategy(distributed_devices).scope():
            yield request.param
    else:
        yield request.param


@pytest.fixture(params=[True, False])
def quantized(request):
    """pytest fixture for running test quantized and non-quantized"""
    with lq_context.quantized_scope(request.param):
        yield request.param


@pytest.fixture(params=["channels_last", "channels_first"])
def data_format(request):
    return request.param
