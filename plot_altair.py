import os
import uuid
from functools import reduce

import numpy as np
import tensorflow as tf

import altair as alt
import larq as lq
import pandas as pd

try:
    tf.enable_eager_execution()
except:
    pass


def calculate_activation(function, x):
    tf_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        activation = function(tf_x)
    return activation.numpy(), tape.gradient(activation, tf_x).numpy()


def html_format(source, language=None, css_class=None, options=None, md=None):
    div_id = f"altair-plot-{uuid.uuid4()}"
    return f"""
        <div id="{ div_id }">
        <script>
          // embed when document is loaded, to ensure vega library is available
          // this works on all modern browsers, except IE8 and older
          document.addEventListener("DOMContentLoaded", function(event) {{
              var opt = {{
                "mode": "vega-lite",
                "renderer": "canvas",
                "actions": false,
              }};
              vegaEmbed('#{ div_id }', '{source}', opt).catch(console.err);
          }}, {{passive: true, once: true}});
        </script>
        </div>
        """


def plot_activation(source, language=None, css_class=None, options=None, md=None):
    function = reduce(getattr, [lq, *source.split(".")])
    x = np.linspace(-2, 2, 500)
    y, dy = calculate_activation(function, x)
    data = pd.DataFrame({"x": x, "y": y, "dy / dx": dy})

    base = alt.Chart(data, width=280, height=180).mark_line().encode(x="x")
    forward = base.encode(y="y").properties(title="Forward pass")
    backward = base.encode(y="dy / dx").properties(title="Backward pass")

    base_path = os.path.join(os.path.dirname(__file__), "docs", "plots", "generated")
    os.makedirs(base_path, exist_ok=True)
    file_name = f"{source}.vg.json"

    (forward | backward).save(os.path.join(base_path, file_name))
    return html_format(f"/plots/generated/{file_name}")
