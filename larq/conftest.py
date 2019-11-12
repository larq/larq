import pytest
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


@pytest.fixture(params=["tf_eager", "tf_keras_eager", "graph"])
def fixture_run_all_keras_modes(request):
    """TODO"""

    if request.param == "graph":
        with context.graph_mode():
            yield

    else:
        with context.eager_mode():
            yield request.param == "tf_keras_eager"
