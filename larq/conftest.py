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
