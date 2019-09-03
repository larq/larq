from functools import reduce
import matplotlib as mpl
import numpy as np
import larq as lq
import tensorflow as tf
from io import StringIO
from scour import scour

mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["font.sans-serif"] = [
    "Roboto",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "sans-serif",
]

import matplotlib.pyplot as plt

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
    ax1.set_title("Forward pass")
    ax2.plot(x, dy, color="#3f51b5")
    ax2.set_title("Backward pass")
    ax2.set_xlabel("x")
    ax2.set_ylabel("dy / dx")
    fig.tight_layout(pad=0.7)
    return fig


def html_format(source, language, css_class, options, md):
    function = reduce(getattr, [lq, *source.split(".")])
    fig = plot(function)
    tmp = StringIO()
    fig.savefig(tmp, format="svg", bbox_inches="tight", pad_inches=0)
    return scour.scourString(tmp.getvalue().replace("DejaVu Sans", "sans-serif"))
