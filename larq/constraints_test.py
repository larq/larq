import pytest
import numpy as np
import tensorflow as tf
import larq as lq


@pytest.mark.parametrize("name", ["weight_clip"])
def test_serialization(name):
    fn = tf.keras.constraints.get(name)
    ref_fn = getattr(lq.constraints, name)()
    assert fn.__class__ == ref_fn.__class__
    config = tf.keras.constraints.serialize(fn)
    fn = tf.keras.constraints.deserialize(config)
    assert fn.__class__ == ref_fn.__class__


def test_clip():
    real_values = np.random.uniform(-2, 2, (3, 3, 32))
    clip_instance = lq.constraints.weight_clip(clip_value=0.75)
    result = clip_instance(tf.keras.backend.variable(real_values))
    result = tf.keras.backend.eval(result)
    np.testing.assert_allclose(result, np.clip(real_values, -0.75, 0.75))
