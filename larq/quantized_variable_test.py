import tensorflow as tf
from tensorflow.python.distribute.values import DistributedVariable

from larq.quantized_variable import create_quantized_variable


def test_inheritance(distribute_scope):
    variable = tf.Variable(3.0)
    quantized_variable = create_quantized_variable(variable)
    assert isinstance(quantized_variable, tf.Variable)
    assert isinstance(quantized_variable, DistributedVariable) is distribute_scope  # type: ignore
