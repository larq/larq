from functools import reduce
import matplotlib
import numpy as np
import larq as lq
import tensorflow as tf
import matplotlib.pyplot as plt
from io import StringIO

matplotlib.use("Agg")
try:
    tf.enable_eager_execution()
except:
    pass


def calculate_activation(function, x):
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        activation = function(tf_x)
    return activation.numpy(), tape.gradient(activation, tf_x).numpy()


def plot(function):
    x = np.linspace(-2, 2, 500)
    y, dy = calculate_activation(function, x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(x, y, color="#3f51b5")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.plot(x, dy, color="#3f51b5")
    ax2.set_xlabel("x")
    ax2.set_ylabel("dy / dx")
    fig.tight_layout(pad=0.7)
    return fig


def html_format(source, language, css_class, options, md):
    fig = plot(reduce(getattr, [lq, *source.split(".")]))
    tmp = StringIO()
    fig.savefig(tmp, format="svg", bbox_inches="tight")
    return tmp.getvalue()
