import numpy as np
import pdb
from terminaltables import AsciiTable
from collections import defaultdict
import itertools

import tensorflow.keras.layers as keras_layers
import larq.layers as lq_layers

__all__ = ["summary"]

op_count_supported_layer_types = [
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
]

mac_layers = [
    lq_layers.QuantConv2D,
    lq_layers.QuantSeparableConv2D,
    lq_layers.QuantDepthwiseConv2D,
    lq_layers.QuantDense,
    keras_layers.Conv2D,
    keras_layers.SeparableConv2D,
    keras_layers.DepthwiseConv2D,
    keras_layers.Dense,
]


def _flatten(lst):
    return list(itertools.chain.from_iterable(lst))


def _bitsize_as_str(bitsize):
    bitsize_names = {8: "byte", 8 * 1024: "kB"}

    try:
        return bitsize_names[bitsize]
    except:
        raise NotImplementedError()


def _format_table_entry(x, units=1):
    if type(x) == str or x == 0 or units == 1:
        return x
    return x / units


def _get_output_shape(layer):
    try:
        return tuple(dim if dim else -1 for dim in layer.output_shape)
    except AttributeError:
        return "multiple"
    except RuntimeError:  # output_shape unknown in Eager mode.
        return "?"


class ParameterProfile:
    bitwidth = None

    def __init__(self, parameter, bitwidth=32):
        self._parameter = parameter
        self.bitwidth = bitwidth

    @property
    def count(self):
        return int(np.prod(self._parameter.shape.as_list()))

    @property
    def memory(self):
        return self.bitwidth * self.count

    @property
    def fp_equivalent_memory(self):
        return 32 * self.count

    @property
    def trainable(self):
        return self._parameter.trainable

    def is_bias(self):
        return "bias" in self._parameter.name


class OperationProfile:
    def __init__(self, n, precision, op_type):
        self.n = n
        self.precision = precision
        self.op_type = op_type


class LayerProfile:
    def __init__(self, layer):
        self._layer = layer
        self.parameter_profiles = []
        self.op_profiles = []

        for weight in layer.weights:
            self.parameter_profiles.append(
                ParameterProfile(weight, self._get_bitwidth(weight))
            )

        if type(layer) in mac_layers:
            # mac layers should have a kernel weight and perhaps a bias
            assert len(self.parameter_profiles) <= 2

            for p in self.parameter_profiles:
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
        return sum([p.memory for p in self.parameter_profiles])

    @property
    def fp_equivalent_memory(self):
        return sum([p.fp_equivalent_memory for p in self.parameter_profiles])

    def parameter_count(self, bitwidth=None, trainable=None):
        count = 0
        for p in self.parameter_profiles:
            if bitwidth and (p.bitwidth != bitwidth):
                continue
            if (trainable is not None) and (p.trainable != trainable):
                continue
            count += p.count
        return count

    def op_count(self, precision=None, op_type=None, escape="?"):
        if type(self._layer) in op_count_supported_layer_types:
            count = 0
            for op in self.op_profiles:
                if precision and (op.precision != precision):
                    continue
                if op_type and (op.op_type != op_type):
                    continue
                count += op.n
            return count
        else:
            return escape

    def input_precision(self, default="-"):
        try:
            return self._layer.input_quantizer.precision
        except:
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
        if len(self.output_shape) == 4:
            return np.prod(self.output_shape[1:2])
        elif len(self.output_shape) == 2:
            return 1
        else:
            raise NotImplementedError()

    @property
    def unique_param_bidtwidths(self):
        return sorted(np.unique([p.bitwidth for p in self.parameter_profiles]))

    @property
    def unique_op_precisions(self):
        return sorted(np.unique([op.precision for op in self.op_profiles]))

    def generate_table_row(self, table_config):
        row = [self._layer.name, self.input_precision(), self.output_shape]
        for i in table_config["param_bidtwidths"]:
            n = self.parameter_count(i)
            n = _format_table_entry(n, table_config["param_units"])
            row.append(n)
        row.append(_format_table_entry(self.memory, table_config["memory_units"]))
        for i in table_config["mac_precisions"]:
            n = self.op_count(i, "mac")
            n = _format_table_entry(n, table_config["mac_units"])
            row.append(n)

        return row

    def _quantized_weights(self):
        try:
            return self._layer.quantized_latent_weights
        except:
            return []

    def _get_bitwidth(self, weight):
        try:
            quantizer = self._layer.quantizers[self._quantized_weights().index(weight)]
            return quantizer.precision
        except:
            return 32


class ModelProfile:
    def __init__(self, model):
        self.layer_profiles = [LayerProfile(l) for l in model.layers]

    @property
    def memory(self):
        return sum([l.memory for l in self.layer_profiles])

    @property
    def fp_equivalent_memory(self):
        return sum([l.fp_equivalent_memory for l in self.layer_profiles])

    def parameter_count(self, bitwidth=None, trainable=None):
        return sum(
            [l.parameter_count(bitwidth, trainable) for l in self.layer_profiles]
        )

    def op_count(self, bitwidth=None, op_type=None):
        return sum([l.op_count(bitwidth, op_type, 0) for l in self.layer_profiles])

    @property
    def unique_param_bidtwidths(self):
        return sorted(
            np.unique(
                _flatten([l.unique_param_bidtwidths for l in self.layer_profiles])
            )
        )

    @property
    def unique_op_precisions(self):
        return sorted(
            np.unique(_flatten([l.unique_op_precisions for l in self.layer_profiles]))
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
            *(
                f"{i}-bit macs\n({_bitsize_as_str(table_config['mac_units'])})"
                for i in table_config["mac_precisions"]
            ),
        ]

    def _generate_table_total(self, table_config):
        row = ["Total", "", ""]
        for i in table_config["param_bidtwidths"]:
            row.append(
                _format_table_entry(
                    self.parameter_count(i), table_config["param_units"]
                )
            )
        row.append(_format_table_entry(self.memory, table_config["memory_units"]))
        for i in table_config["mac_precisions"]:
            row.append(
                _format_table_entry(self.op_count(i, "mac"), table_config["mac_units"])
            )
        return row

    def generate_table(self, include_macs=True):
        table_config = {
            "param_bidtwidths": self.unique_param_bidtwidths,
            "mac_precisions": self.unique_op_precisions if include_macs else [],
            "param_units": 1,
            "memory_units": 8 * 1024,
            "mac_units": 8 * 1024,
        }

        table = []

        table.append(self._generate_table_header(table_config))

        for lp in self.layer_profiles:
            table.append(lp.generate_table_row(table_config))

        table.append(self._generate_table_total(table_config))

        return table

    def generate_summary(self, include_macs=True):
        summary = [
            ["Total params", self.parameter_count()],
            ["Trainable params", self.parameter_count(trainable=True)],
            ["Non-trainable params", self.parameter_count(trainable=False)],
            ["Model size:", f"{self.memory / (8*1024*1024):.2f} MB"],
            [
                "Float-32 Equivalent",
                f"{self.fp_equivalent_memory / (8*1024*1024):.2f} MB",
            ],
            ["Compression Ratio of Memory", self.memory / self.fp_equivalent_memory],
        ]

        if include_macs:
            binarization_ratio = self.op_count(1, "mac") / self.op_count(op_type="mac")
            summary.extend(
                [
                    ["Number of MACs", self.op_count(op_type="mac")],
                    ["Ratio of MACs that are binarized", f"{binarization_ratio:.4f}"],
                ]
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

    if print_fn is None:
        print_fn = print

    model_profile = ModelProfile(model)
    print_fn(
        LayersTable(model_profile.generate_table(), title=f"{model.name} stats").table
    )
    print_fn(
        SummaryTable(
            model_profile.generate_summary(), title=f"{model.name} summary"
        ).table
    )
