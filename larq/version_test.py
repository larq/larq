import larq


def test_version():
    assert hasattr(larq, "__version__") and "." in larq.__version__
