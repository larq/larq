import itertools
from dataclasses import dataclass

import numpy as np
import tensorflow.keras.layers as keras_layers
from terminaltables import AsciiTable

import larq.layers as lq_layers

__all__ = ["summary"]

op_count_supported_layer_types = (
    lq_layers.QuantConv2D,
    lq_layers.QuantSeparableConv2D,
    lq_layers.QuantDepthwiseConv2D,
    lq_layers.QuantDense,
    keras_layers.Conv2D,
    keras_layers.SeparableConv2D,
    keras_layers.DepthwiseConv2D,
    keras_layers.Dense,
    keras_layers.Flatten,
    keras_layers.BatchNormalization,
    keras_layers.MaxPool2D,
    keras_layers.AveragePooling2D,
)

mac_containing_layers = (
    lq_layers.QuantConv2D,
    lq_layers.QuantSeparableConv2D,
    lq_layers.QuantDepthwiseConv2D,
    lq_layers.QuantDense,
    keras_layers.Conv2D,
    keras_layers.SeparableConv2D,
    keras_layers.DepthwiseConv2D,
    keras_layers.Dense,
)


def _flatten(lst):
    return list(itertools.chain.from_iterable(lst))


def _bitsize_as_str(bitsize):
    bitsize_names = {8: "byte", 8 * 1024: "kB"}

    try:
        return bitsize_names[bitsize]
    except KeyError:
        raise NotImplementedError()


def _number_as_readable_str(num):
    # The initial rounding here is necessary so that e.g. `999000` gets
    # formatted as `1.000 M` rather than `1000 k`
    num = float(f"{num:.3g}")

    # For numbers less than 1000, output them directly, stripping any trailing
    # zeros and decimal places.
    if num < 1000:
        return str(num).rstrip("0").rstrip(".")

    # For numbers that are at least 1000 trillion (1 quadrillion) format with
    # scientific notation (3 s.f. = 2 d.p. in scientific notation).
    if num >= 1e15:
        return f"{num:#.2E}"

    # Count the magnitude.
    magnitude = 0
    while abs(num) >= 1000 and magnitude < 4:
        magnitude += 1
        num /= 1000.0

    # ':#.3g' formats the number with 3 significant figures, without stripping
    # trailing zeros.
    num = f"{num:#.3g}".rstrip(".")
    unit = ["", " k", " M", " B", " T"][magnitude]
    return num + unit


def _format_table_entry(x, units=1):
    try:
        assert not np.isnan(x)
        if type(x) == str or x == 0 or units == 1:
            return x
        return x / units
    except Exception:
        return "?"


def _get_output_shape(layer):
    try:
        return tuple(dim if dim else -1 for dim in layer.output_shape)
    except AttributeError:
        return "multiple"
    except RuntimeError:  # output_shape unknown in Eager mode.
        return "?"


class WeightProfile:
    def __init__(self, weight, bitwidth=32, trainable=True):
        self._weight = weight
        self.bitwidth = bitwidth
        self.trainable = trainable

    @property
    def count(self):
        return int(np.prod(self._weight.shape.as_list()))

    @property
    def memory(self):
        return self.bitwidth * self.count

    @property
    def fp_equivalent_memory(self):
        return 32 * self.count

    def is_bias(self):
        return "bias" in self._weight.name


@dataclass
class OperationProfile:
    n: int
    precision: int
    op_type: str


class LayerProfile:
    def __init__(self, layer):
        self._layer = layer
        self.weight_profiles = [
            WeightProfile(
                weight,
                self._get_bitwidth(weight),
                trainable=any(weight is w for w in layer.trainable_weights),
            )
            for weight in layer.weights
        ]

        self.op_profiles = []

        if isinstance(layer, mac_containing_layers):
            for p in self.weight_profiles:
                if not p.is_bias():
                    self.op_profiles.append(
                        OperationProfile(
                            n=p.count * self.output_pixels,
                            precision=max(self.input_precision(32), p.bitwidth),
                            op_type="mac",
                        )
                    )

    @property
    def memory(self):
        return sum(p.memory for p in self.weight_profiles)

    @property
    def fp_equivalent_memory(self):
        return sum(p.fp_equivalent_memory for p in self.weight_profiles)

    def weight_count(self, bitwidth=None, trainable=None):
        count = 0
        for p in self.weight_profiles:
            if (bitwidth is None or p.bitwidth == bitwidth) and (
                trainable is None or p.trainable == trainable
            ):
                count += p.count
        return count

    def op_count(self, op_type=None, precision=None, escape="?"):
        if op_type != "mac":
            raise ValueError("Currently only counting of MAC-operations is supported.")

        if isinstance(self._layer, op_count_supported_layer_types):
            count = 0
            for op in self.op_profiles:
                if (precision is None or op.precision == precision) and (
                    op_type is None or op.op_type == op_type
                ):
                    count += op.n
            return count
        else:
            return escape

    def input_precision(self, default="-"):
        try:
            return self._layer.input_quantizer.precision
        except AttributeError:
            return default

    @property
    def output_shape(self):
        try:
            return tuple(dim if dim else -1 for dim in self._layer.output_shape)
        except AttributeError:
            return "multiple"
        except RuntimeError:  # output_shape unknown in Eager mode.
            return "?"

    @property
    def output_pixels(self):
        """Number of pixels for a single feature map (1 for fully connected layers)."""
        if len(self.output_shape) == 4:
            return int(np.prod(self.output_shape[1:3]))
        elif len(self.output_shape) == 2:
            return 1
        else:
            raise NotImplementedError()

    @property
    def unique_param_bidtwidths(self):
        return sorted(set([p.bitwidth for p in self.weight_profiles]))

    @property
    def unique_op_precisions(self):
        return sorted(set([op.precision for op in self.op_profiles]))

    def generate_table_row(self, table_config):
        row = [self._layer.name, self.input_precision(), self.output_shape]
        for i in table_config["param_bidtwidths"]:
            n = self.weight_count(i)
            n = _format_table_entry(n, table_config["param_units"])
            row.append(n)
        row.append(_format_table_entry(self.memory, table_config["memory_units"]))
        for i in table_config["mac_precisions"]:
            n = self.op_count("mac", i)
            n = _format_table_entry(n, table_config["mac_units"])
            row.append(n)

        return row

    def _get_bitwidth(self, weight):
        try:
            for quantizer, quantized_weight in zip(
                self._layer.quantizers, self._layer.quantized_latent_weights
            ):
                if quantized_weight is weight:
                    return quantizer.precision
        except AttributeError:
            pass
        return 32


class ModelProfile(LayerProfile):
    def __init__(self, model):
        self.layer_profiles = [LayerProfile(l) for l in model.layers]

    @property
    def memory(self):
        return sum(l.memory for l in self.layer_profiles)

    @property
    def fp_equivalent_memory(self):
        return sum(l.fp_equivalent_memory for l in self.layer_profiles)

    def weight_count(self, bitwidth=None, trainable=None):
        return sum(l.weight_count(bitwidth, trainable) for l in self.layer_profiles)

    def op_count(self, op_type=None, bitwidth=None):
        return sum(l.op_count(op_type, bitwidth, 0) for l in self.layer_profiles)

    @property
    def unique_param_bidtwidths(self):
        return sorted(
            set(_flatten(l.unique_param_bidtwidths for l in self.layer_profiles))
        )

    @property
    def unique_op_precisions(self):
        return sorted(
            set(_flatten(l.unique_op_precisions for l in self.layer_profiles))
        )

    def _generate_table_header(self, table_config):
        return [
            "Layer",
            "Input prec.\n(bit)",
            "Outputs",
            *(
                f"# {i}-bit\nx {table_config['param_units']}"
                for i in table_config["param_bidtwidths"]
            ),
            f"Memory\n({_bitsize_as_str(table_config['memory_units'])})",
            *(f"{i}-bit MACs" for i in table_config["mac_precisions"]),
        ]

    def _generate_table_total(self, table_config):
        row = ["Total", "", ""]
        for i in table_config["param_bidtwidths"]:
            row.append(
                _format_table_entry(self.weight_count(i), table_config["param_units"])
            )
        row.append(_format_table_entry(self.memory, table_config["memory_units"]))
        for i in table_config["mac_precisions"]:
            row.append(
                _format_table_entry(self.op_count("mac", i), table_config["mac_units"])
            )
        return row

    def generate_table(self, include_macs=True):
        table_config = {
            "param_bidtwidths": self.unique_param_bidtwidths,
            "mac_precisions": self.unique_op_precisions if include_macs else [],
            "param_units": 1,
            "memory_units": 8 * 1024,
            "mac_units": 1,
        }

        table = []

        table.append(self._generate_table_header(table_config))

        for lp in self.layer_profiles:
            table.append(lp.generate_table_row(table_config))

        table.append(self._generate_table_total(table_config))

        return table

    def generate_summary(self, include_macs=True):
        summary = [
            ["Total params", _number_as_readable_str(self.weight_count())],
            [
                "Trainable params",
                _number_as_readable_str(self.weight_count(trainable=True)),
            ],
            [
                "Non-trainable params",
                _number_as_readable_str(self.weight_count(trainable=False)),
            ],
            ["Model size:", f"{self.memory / (8*1024*1024):.2f} MB"],
            [
                "Float-32 Equivalent",
                f"{self.fp_equivalent_memory / (8*1024*1024):.2f} MB",
            ],
            [
                "Compression Ratio of Memory",
                self.memory / max(1e-8, self.fp_equivalent_memory),
            ],
        ]

        if include_macs:
            binarization_ratio = self.op_count("mac", 1) / max(
                1, self.op_count(op_type="mac")
            )
            ternarization_ratio = self.op_count("mac", 2) / max(
                1, self.op_count(op_type="mac")
            )
            summary.append(
                [
                    "Number of MACs",
                    _number_as_readable_str(self.op_count(op_type="mac")),
                ]
            )
            if binarization_ratio > 0:
                summary.append(
                    ["Ratio of MACs that are binarized", f"{binarization_ratio:.4f}"]
                )
            if ternarization_ratio > 0:
                summary.append(
                    ["Ratio of MACs that are ternarized", f"{ternarization_ratio:.4f}"]
                )

        return summary


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


def summary(model, print_fn=None, include_macs=True):
    """Prints a string summary of the network.

    The summary includes the following information per layer:

    - input precision,
    - output dimension,
    - weight count (broken down by bidtwidth),
    - memory footprint in kilobytes (`8*1024` 1-bit weights = 1 kB),
    - number of multiply-accumulate (MAC) operations broken down by precision (*optional & expermental*).

    A single MAC operation contains both a multiplication and an addition. The precision
    of a MAC operation is defined as the maximum bitwidth of its inputs.

    Additionally, the following overall statistics for the model are supplied:

    - total number of weights,
    - total number of trainable weights,
    - total number of non-trainable weights,
    - model size,
    - float-32 equivalent size: memory footprint if all weights were 32 bit,
    - compression ratio achieved by quantizing weights,
    - total number of MAC operations,
    - ratio of MAC operations that is binarized and can be accelated with XNOR-gates.

    # Arguments
    model: `tf.keras` model instance.
    print_fn: Print function to use. Defaults to `print`. You can set it to a custom
        function in order to capture the string summary.
    include_macs: whether or not to include the number of MAC-operations in the summary.

    # Raises
    ValueError: if called before the model is built.
    """

    if not model.built:
        raise ValueError(
            "This model has not yet been built. Build the model first by calling "
            "`model.build()` or calling `model.fit()` with some data, or specify an "
            "`input_shape` argument in the first layer(s) for automatic build."
        )

    if print_fn is None:
        print_fn = print

    model_profile = ModelProfile(model)
    print_fn(
        LayersTable(model_profile.generate_table(), title=f"{model.name} stats").table
    )
    print_fn(
        SummaryTable(
            model_profile.generate_summary(include_macs), title=f"{model.name} summary"
        ).table
    )
