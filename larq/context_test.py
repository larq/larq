import pytest

from larq import context


def test_scope():
    assert context.get_training_metrics() == set()
    with context.metrics_scope(["flip_ratio"]):
        assert context.get_training_metrics() == {"flip_ratio"}
    assert context.get_training_metrics() == set()
    with pytest.raises(ValueError, match=r".*unknown_metric.*"):
        with context.metrics_scope(["flip_ratio", "unknown_metric"]):
            pass
