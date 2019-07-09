import numpy as np
from terminaltables import AsciiTable


def sanitize_table(table_data):
    return [[f"{v:.2f}" if type(v) == float else v for v in row] for row in table_data]


class LayersTable(AsciiTable):
    def __init__(self, table_data, title=None):
        super().__init__(sanitize_table(table_data), title=title)
        self.inner_column_border = False
        self.justify_columns = {
            i: "left" if i == 0 else "right" for i in range(len(table_data[0]))
        }
        self.inner_footing_row_border = True
        self.inner_heading_row_border = True


class SummaryTable(AsciiTable):
    def __init__(self, table_data, title=None):
        super().__init__(sanitize_table(table_data), title=title)
        self.inner_column_border = False
        self.inner_heading_row_border = False


def _count_params(weights, ignore=[]):
    """Count the total number of scalars composing the weights.

    # Arguments
    weights: An iterable containing the weights on which to compute params
    ignore: A list of weights to ignore

    # Returns
    The total number of scalars composing the weights
    """
    return int(sum(np.prod(w.shape.as_list()) for w in weights if w not in ignore))


def _get_output_shape(layer):
    try:
        return tuple(dim if dim else -1 for dim in layer.output_shape)
    except AttributeError:
        return "multiple"
    except RuntimeError:  # output_shape unknown in Eager mode.
        return "?"


def _compute_memory(layer_stat):
    mem_in_bits = sum(precision * number for precision, number in layer_stat.items())
    return _bit_to_kB(mem_in_bits)


def _parse_params(layer, ignore=[]):
    if hasattr(layer, "quantized_latent_weights"):
        params = {}
        for quantizer, weight in zip(layer.quantizers, layer.quantized_latent_weights):
            if weight not in ignore:
                precision = getattr(quantizer, "precision", 32)
                params[precision] = params.get(precision, 0) + int(
                    np.prod(weight.shape.as_list())
                )
        for weight in layer.weights:
            if weight not in layer.quantized_latent_weights and weight not in ignore:
                params[32] = params.get(32, 0) + int(np.prod(weight.shape.as_list()))
        return params
    return {32: layer.count_params()}


def _sum_params(layer_stats):
    params = {}
    for layer_stat in layer_stats:
        for key, value in layer_stat.items():
            params[key] = params.get(key, 0) + value
    return params


def _row_from_stats(stats, summed_stats):
    return (stats.get(key, 0) for key in sorted(summed_stats))


def _generate_table(model, ignore=[]):
    layer_stats = [_parse_params(l, ignore=ignore) for l in model.layers]
    summed_stat = _sum_params(layer_stats)

    table = [
        [
            "Layer",
            "Outputs",
            *(f"# {i}-bit" for i in sorted(summed_stat)),
            "Memory (kB)",
        ]
    ]
    for layer, stats in zip(model.layers, layer_stats):
        table.append(
            [
                layer.name,
                _get_output_shape(layer),
                *_row_from_stats(stats, summed_stat),
                _compute_memory(stats),
            ]
        )
    table.append(
        [
            "Total",
            "",
            *_row_from_stats(summed_stat, summed_stat),
            _compute_memory(summed_stat),
        ]
    )
    return table


def _bit_to_kB(bit_value):
    return bit_value / 8 / 1024


def summary(model, tablefmt="simple", print_fn=None):
    """Prints a string summary of the network.

    # Arguments
    model: `tf.keras` model instance.
    tablefmt: Supported table formats are: `fancy_grid`, `github`, `grid`, `html`,
        `jira`, `latex`, `latex_booktabs`, `latex_raw`, `mediawiki`, `moinmoin`,
        `orgtbl`, `pipe`, `plain`, `presto`, `psql`, `rst`, `simple`, `textile`,
        `tsv`, `youtrac`.
    print_fn: Print function to use. Defaults to `print`. You can set it to a custom
        function in order to capture the string summary.

    # Raises
    ValueError: if called before the model is built.
    """

    if not model.built:
        raise ValueError(
            "This model has not yet been built. Build the model first by calling "
            "`model.build()` or calling `model.fit()` with some data, or specify an "
            "`input_shape` argument in the first layer(s) for automatic build."
        )

    metrics_weights = [weight for metric in model.metrics for weight in metric.weights]
    table = _generate_table(model, ignore=metrics_weights)

    model._check_trainable_weights_consistency()
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = _count_params(
            model._collected_trainable_weights, ignore=metrics_weights
        )
    else:
        trainable_count = _count_params(model.trainable_weights, ignore=metrics_weights)
    non_trainable_count = _count_params(
        model.non_trainable_weights, ignore=metrics_weights
    )

    if print_fn is None:
        print_fn = print

    total_params = trainable_count + non_trainable_count
    float_32_memory_equiv = _bit_to_kB(total_params * 32)
    compression_ratio = float_32_memory_equiv / table[-1][-1]

    summary_table = [
        ["Total params", total_params],
        ["Trainable params", trainable_count],
        ["Non-trainable params", non_trainable_count],
        ["Float-32 Equivalent", f"{float_32_memory_equiv / 1024:.2f} MB"],
        ["Compression of Memory", compression_ratio],
    ]

    print_fn(LayersTable(table, title=f"{model.name} stats").table)
    print_fn(SummaryTable(summary_table, title=f"{model.name} summary").table)
