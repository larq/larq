import numpy as np
from terminaltables import AsciiTable


class LayersTable(AsciiTable):
    def __init__(self, table_data, title=None):
        super().__init__(table_data, title=title)
        self.inner_column_border = False
        self.inner_footing_row_border = True
        self.inner_heading_row_border = True


class SummaryTable(AsciiTable):
    def __init__(self, table_data, title=None):
        super().__init__(table_data, title=title)
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


def _count_binarized_weights(layer):
    if hasattr(layer, "quantized_latent_weights"):
        return _count_params(layer.quantized_latent_weights)
    return 0


def _count_fp_weights(layer, ignore=[]):
    ignored_weights = getattr(layer, "quantized_latent_weights", []) + ignore
    return _count_params(layer.weights, ignored_weights)


def _bit_to_kB(bit_value):
    return bit_value / 8 / 1024


def _memory_weights(layer, ignore=[]):
    num_fp_params = _count_fp_weights(layer, ignore=ignore)
    num_binarized_params = _count_binarized_weights(layer)
    fp32 = 32  # Multiply float32 params by 32 to get bit value
    total_layer_mem_in_bits = (num_fp_params * fp32) + (num_binarized_params)
    return _bit_to_kB(total_layer_mem_in_bits)


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
    table = [
        [
            layer.name,
            _get_output_shape(layer),
            _count_binarized_weights(layer),
            _count_fp_weights(layer, ignore=metrics_weights),
            _memory_weights(layer, ignore=metrics_weights),
        ]
        for layer in model.layers
    ]

    amount_binarized = sum(r[2] for r in table)
    amount_full_precision = sum(r[3] for r in table)
    total_memory = sum(r[4] for r in table)
    table.insert(0, ["Layer", "Outputs", "# 1-bit", "# 32-bit", "Memory (kB)"])
    table.append(["Total", "", amount_binarized, amount_full_precision, total_memory])

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

    float32_equiv = _bit_to_kB((amount_binarized + amount_full_precision) * 32)
    compression_ratio = float32_equiv / total_memory

    summary_table = [
        ["Total params", trainable_count + non_trainable_count],
        ["Trainable params", trainable_count],
        ["Non-trainable params", non_trainable_count],
        ["Float-32 Equivalent", f"{float32_equiv / 1024:.2f} MB"],
        ["Compression of Memory", compression_ratio],
    ]

    print_fn(LayersTable(table, title=f"{model.name} stats").table)
    print_fn(SummaryTable(summary_table, title=f"{model.name} summary").table)
    print_fn()
