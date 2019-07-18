import numpy as np
import pdb
from terminaltables import AsciiTable
from collections import defaultdict

from tensorflow.keras.layers import (
    Conv2D,
    SeparableConv2D,
    DepthwiseConv2D,
    Dense,
    BatchNormalization,
    MaxPool2D,
    AveragePooling2D,
    Flatten,
)
from larq.layers import (
    QuantConv2D,
    QuantSeparableConv2D,
    QuantDepthwiseConv2D,
    QuantDense,
)

__all__ = ["summary"]

mac_count_supported_layer_types = [
    QuantConv2D,
    QuantSeparableConv2D,
    QuantDepthwiseConv2D,
    QuantDense,
    Conv2D,
    SeparableConv2D,
    DepthwiseConv2D,
    Dense,
    Flatten,
    BatchNormalization,
    MaxPool2D,
    AveragePooling2D,
]

mac_layers = [
    QuantConv2D,
    QuantSeparableConv2D,
    QuantDepthwiseConv2D,
    QuantDense,
    Conv2D,
    SeparableConv2D,
    DepthwiseConv2D,
    Dense,
]


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


def _count_params(weights):
    """Count the total number of scalars composing the weights.

    # Arguments
    weights: An iterable containing the weights on which to compute params

    # Returns
    The total number of scalars composing the weights
    """
    return int(sum(np.prod(w.shape.as_list()) for w in weights))


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

def _count_weights(weight_var):
    return int(np.prod(weight_var.shape.as_list()))

def _parse_params(layer, include_bias_params=True):
    params = defaultdict(int)
    ignored_param_count = 0

    if hasattr(layer, "quantized_latent_weights"):
        quantized_latent_weights = layer.quantized_latent_weights
    else:
        quantized_latent_weights = []

    if len(quantized_latent_weights) != 0:
        for quantizer, weight in zip(layer.quantizers, quantized_latent_weights):
            if not include_bias_params and 'bias' in weight.name:
                continue
            precision = getattr(quantizer, "precision", 32)
            params[precision] = params.get(precision, 0) + _count_weights(weight)

    for weight in layer.weights:
        if not include_bias_params and 'bias' in weight.name:
            ignored_param_count += _count_weights(weight)
            continue
        if weight not in quantized_latent_weights:
            params[32] = params.get(32, 0) + _count_weights(weight)

    # sanity check
    assert sum(params.values()) + ignored_param_count == layer.count_params()

    return params

def _count_layer_params(layer, bits=None, include_bias_params=True):
        count = 0
        params = _parse_params(layer, include_bias_params=include_bias_params)
        if bits:
            return params[bits]
        else:
            return sum(params.values())

def _sum_params(layer_stats):
    params = {}
    for layer_stat in layer_stats:
        for key, value in layer_stat.items():
            params[key] = params.get(key, 0) + value
    return params

def _row_from_stats(stats, summed_stats):
    return (stats.get(key, 0) for key in sorted(summed_stats))


def _get_input_precision(layer):
    try:
        return layer.input_quantizer.precision
    except:
        return "-"


def _get_pixel_count(layer):
    output_shape = _get_output_shape(layer)
    if len(output_shape) == 4:
        return np.prod(output_shape[1:2])
    elif len(output_shape) == 2:
        return 1
    else:
        raise NotImplementedError()


def _get_binary_macs(layer):
    if type(layer) not in mac_count_supported_layer_types:
        return "?"
    if type(layer) not in mac_layers:
        return 0

    try:
        input_precision = layer.input_quantizer.precision
    except:
        return 0

    if input_precision == 1:
        n_params = _count_layer_params(layer, bits=1, include_bias_params=False)
        pixels = _get_pixel_count(layer)
        return _bit_to_kB(n_params * pixels)
    return 0


def _get_total_macs(layer):
    if type(layer) not in mac_count_supported_layer_types:
        return "?"
    if type(layer) not in mac_layers:
        return 0

    try:
        pixels = _get_pixel_count(layer)
        params = _count_layer_params(layer, include_bias_params=False)
        return _bit_to_kB(pixels * params)
    except:
        return "?"


def _generate_table(model):
    layer_stats = [_parse_params(l) for l in model.layers]
    summed_stat = _sum_params(layer_stats)

    bin_macs = 0
    total_macs = 0

    table = [
        [
            "Layer",
            "Input prec.\n(bit)",
            "Outputs",
            *(f"# {i}-bit" for i in sorted(summed_stat)),
            "Memory\n(kB)",
            "bmacs (kB)",
            "macs (kB)",
        ]
    ]
    for layer, stats in zip(model.layers, layer_stats):
        layer_bin_macs = _get_binary_macs(layer)
        layer_total_macs = _get_total_macs(layer)
        table.append(
            [
                layer.name,
                _get_input_precision(layer),
                _get_output_shape(layer),
                *_row_from_stats(stats, summed_stat),
                _compute_memory(stats),
                layer_bin_macs,
                layer_total_macs,
            ]
        )

        if type(layer_bin_macs) is not str:
            bin_macs += layer_bin_macs
        if type(layer_total_macs) is not str:
            total_macs += layer_total_macs

    table.append(
        [
            "Total",
            "",
            "",
            *_row_from_stats(summed_stat, summed_stat),
            _compute_memory(summed_stat),
            bin_macs,
            total_macs,
        ]
    )
    return table


def _bit_to_kB(bit_value):
    return bit_value / 8 / 1024


def summary(model, print_fn=None):
    """Prints a string summary of the network.

    # Arguments
    model: `tf.keras` model instance.
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

    table = _generate_table(model)

    model._check_trainable_weights_consistency()
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = _count_params(model._collected_trainable_weights)
    else:
        trainable_count = _count_params(model.trainable_weights)
    non_trainable_count = _count_params(model.non_trainable_weights)

    if print_fn is None:
        print_fn = print

    total_params = trainable_count + non_trainable_count
    float_32_memory_equiv = _bit_to_kB(total_params * 32)
    compression_ratio = float_32_memory_equiv / table[-1][4]
    binarization_ratio = table[-1][-2]/table[-1][-1]

    summary_table = [
        ["Total params", total_params],
        ["Trainable params", trainable_count],
        ["Non-trainable params", non_trainable_count],
        ["Float-32 Equivalent", f"{float_32_memory_equiv / 1024:.2f} MB"],
        ["Compression of Memory", compression_ratio],
        ["Ratio of MACS that are binarized", binarization_ratio]
    ]

    print_fn(LayersTable(table, title=f"{model.name} stats").table)
    print_fn(SummaryTable(summary_table, title=f"{model.name} summary").table)
