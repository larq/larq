import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, TypeVar, Union

import numpy as np
import tensorflow as tf
from terminaltables import AsciiTable

from larq import layers as lq_layers
from larq.utils import memory_as_readable_str

__all__ = ["summary"]

mac_containing_layers = (
    lq_layers.QuantConv2D,
    lq_layers.QuantSeparableConv2D,
    lq_layers.QuantDepthwiseConv2D,
    lq_layers.QuantDense,
    tf.keras.layers.Conv2D,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.Dense,
    lq_layers.QuantConv1D,
    lq_layers.QuantSeparableConv1D,
    tf.keras.layers.Conv1D,
    tf.keras.layers.SeparableConv1D,
)

op_count_supported_layer_types = (
    tf.keras.layers.Flatten,
    tf.keras.layers.BatchNormalization,
    tf.keras.layers.MaxPool2D,
    tf.keras.layers.AveragePooling2D,
    tf.keras.layers.MaxPool1D,
    tf.keras.layers.AveragePooling1D,
    *mac_containing_layers,
)

T = TypeVar("T")


def _flatten(lst: Iterator[Iterator[T]]) -> Sequence[T]:
    return list(itertools.chain.from_iterable(lst))


def _bitsize_as_str(bitsize: int) -> str:
    bitsize_names = {8: "byte", 8 * 1024: "kB"}

    try:
        return bitsize_names[bitsize]
    except KeyError:
        raise NotImplementedError()


def _number_as_readable_str(num: float) -> str:
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
        return f"{num:.2E}"

    # Count the magnitude.
    magnitude = 0
    while abs(num) >= 1000 and magnitude < 4:
        magnitude += 1
        num /= 1000.0

    # ':.3g' formats the number with 3 significant figures, without stripping trailing
    # zeros.
    num = f"{num:.3g}".rstrip(".")
    unit = ["", " k", " M", " B", " T"][magnitude]
    return num + unit


def _format_table_entry(x: float, units: int = 1) -> Union[float, str]:
    try:
        assert not np.isnan(x)
        if type(x) == str or x == 0 or units == 1:
            return x
        return x / units
    except Exception:
        return "?"


def _normalize_shape(shape):
    return tuple(dim if dim else -1 for dim in shape)


class WeightProfile:
    def __init__(self, weight, trainable: bool = True):
        self._weight = weight
        self.bitwidth = getattr(weight, "precision", 32)
        self.trainable = trainable

    @property
    def count(self) -> int:
        return int(np.prod(self._weight.shape.as_list()))

    @property
    def memory(self) -> int:
        return self.bitwidth * self.count

    @property
    def fp_equivalent_memory(self) -> int:
        return 32 * self.count

    @property
    def int8_fp_weights_memory(self) -> int:
        """Count any 32- or 16-bit weights as 8 bits instead."""

        if self.bitwidth > 8:
            return self.count * 8
        return self.bitwidth * self.count

    def is_bias(self) -> bool:
        return "bias" in self._weight.name


@dataclass
class OperationProfile:
    n: int
    precision: int
    op_type: str


class LayerProfile:
    def __init__(self, layer: tf.keras.layers.Layer):
        self._layer = layer

        weights = layer.weights
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            fused_pairs = [("beta", "moving_mean"), ("gamma", "moving_variance")]
            for pair in fused_pairs:
                names = [w.name.split("/")[-1].replace(":0", "") for w in weights]
                if pair[0] in names and pair[1] in names:
                    weights.pop(names.index(pair[0]))

        self.weight_profiles = [
            WeightProfile(
                weight,
                trainable=any(weight is w for w in layer.trainable_weights),
            )
            for weight in weights
        ]

        self.op_profiles = []

        if isinstance(layer, mac_containing_layers) and self.output_pixels:
            for p in self.weight_profiles:
                if not p.is_bias():
                    self.op_profiles.append(
                        OperationProfile(
                            n=p.count * self.output_pixels,
                            precision=max(self.input_precision or 32, p.bitwidth),
                            op_type="mac",
                        )
                    )

    @property
    def memory(self) -> int:
        return sum(p.memory for p in self.weight_profiles)

    @property
    def int8_fp_weights_memory(self) -> int:
        return sum(p.int8_fp_weights_memory for p in self.weight_profiles)

    @property
    def fp_equivalent_memory(self) -> int:
        return sum(p.fp_equivalent_memory for p in self.weight_profiles)

    def weight_count(
        self, bitwidth: Optional[int] = None, trainable: Optional[bool] = None
    ) -> int:
        count = 0
        for p in self.weight_profiles:
            if (bitwidth is None or p.bitwidth == bitwidth) and (
                trainable is None or p.trainable == trainable
            ):
                count += p.count
        return count

    def op_count(
        self, op_type: Optional[str] = None, precision: Optional[int] = None
    ) -> Optional[int]:
        if op_type != "mac":
            raise ValueError("Currently only counting of MAC-operations is supported.")

        if (
            isinstance(self._layer, op_count_supported_layer_types)
            and self.output_pixels
        ):
            count = 0
            for op in self.op_profiles:
                if (precision is None or op.precision == precision) and (
                    op_type is None or op.op_type == op_type
                ):
                    count += op.n
            return count
        return None

    @property
    def input_precision(self) -> Optional[int]:
        try:
            return self._layer.input_quantizer.precision
        except AttributeError:
            return None

    @property
    def output_shape(self) -> Optional[Sequence[int]]:
        try:
            output_shape = self._layer.output_shape
            if isinstance(output_shape, list):
                if len(output_shape) == 1:
                    return _normalize_shape(output_shape[0])
                return [_normalize_shape(shape) for shape in output_shape]
            return _normalize_shape(output_shape)
        except AttributeError:
            return None

    @property
    def output_shape_str(self) -> str:
        try:
            return str(self.output_shape or "multiple")
        except RuntimeError:
            return "?"

    @property
    def output_pixels(self) -> Optional[int]:
        """Number of pixels for a single feature map (1 for fully connected layers)."""
        if not self.output_shape:
            return None
        if len(self.output_shape) == 4:
            return int(np.prod(self.output_shape[1:3]))
        if len(self.output_shape) == 3:
            return self.output_shape[1]
        if len(self.output_shape) == 2:
            return 1
        raise NotImplementedError()

    @property
    def unique_param_bidtwidths(self) -> Sequence[int]:
        return sorted(set([p.bitwidth for p in self.weight_profiles]))

    @property
    def unique_op_precisions(self) -> Sequence[int]:
        return sorted(set([op.precision for op in self.op_profiles]))

    def generate_table_row(
        self, table_config: Mapping[str, Any]
    ) -> Sequence[Union[str, float]]:
        row = [self._layer.name, self.input_precision or "-", self.output_shape_str]
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


class ModelProfile(LayerProfile):
    def __init__(self, model: tf.keras.models.Model):
        self.layer_profiles = [LayerProfile(layer) for layer in model.layers]

    @property
    def memory(self) -> int:
        return sum(lp.memory for lp in self.layer_profiles)

    @property
    def int8_fp_weights_memory(self) -> int:
        return sum(lp.int8_fp_weights_memory for lp in self.layer_profiles)

    @property
    def fp_equivalent_memory(self) -> int:
        return sum(lp.fp_equivalent_memory for lp in self.layer_profiles)

    def weight_count(
        self, bitwidth: Optional[int] = None, trainable: Optional[bool] = None
    ) -> int:
        return sum(lp.weight_count(bitwidth, trainable) for lp in self.layer_profiles)

    def op_count(
        self, op_type: Optional[str] = None, bitwidth: Optional[int] = None
    ) -> int:
        return sum(lp.op_count(op_type, bitwidth) or 0 for lp in self.layer_profiles)

    @property
    def unique_param_bidtwidths(self) -> Sequence[int]:
        return sorted(
            set(_flatten(lp.unique_param_bidtwidths for lp in self.layer_profiles))
        )

    @property
    def unique_op_precisions(self) -> Sequence[int]:
        return sorted(
            set(_flatten(lp.unique_op_precisions for lp in self.layer_profiles))
        )

    def _generate_table_header(self, table_config: Mapping[str, Any]) -> Sequence[str]:
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

    def _generate_table_total(
        self, table_config: Mapping[str, Any]
    ) -> Sequence[Union[float, str]]:
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

    def generate_table(
        self, include_macs: bool = True
    ) -> Sequence[Sequence[Union[float, str]]]:
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

    def generate_summary(
        self, include_macs: bool = True
    ) -> Sequence[Sequence[Union[str, float]]]:
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
            ["Model size", memory_as_readable_str(self.memory)],
            [
                "Model size (8-bit FP weights)",
                memory_as_readable_str(self.int8_fp_weights_memory),
            ],
            ["Float-32 Equivalent", memory_as_readable_str(self.fp_equivalent_memory)],
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


def sanitize_table(table_data: Sequence[Sequence[Any]]) -> Sequence[Sequence[str]]:
    return [
        [f"{v:.2f}" if type(v) == float else str(v) for v in row] for row in table_data
    ]


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


def summary(
    model: tf.keras.models.Model,
    print_fn: Callable[[str], Any] = None,
    include_macs: bool = True,
) -> None:
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
    - model size (8-bit FP weights): memory footprint if FP weights were 8 bit,
    - float-32 equivalent size: memory footprint if all weights were 32 bit,
    - compression ratio achieved by quantizing weights,
    - total number of MAC operations,
    - ratio of MAC operations that is binarized and can be accelated with XNOR-gates.

    # Arguments
        model: model instance.
        print_fn: Print function to use. Defaults to `print`. You can set it to a custom
            function in order to capture the string summary.
        include_macs: whether or not to include the number of MAC-operations in the
            summary.

    # Raises
        ValueError: if called before the model is built.
    """

    if not model.built:
        raise ValueError(
            "This model has not yet been built. Build the model first by calling "
            "`model.build()` or calling `model.fit()` with some data, or specify an "
            "`input_shape` argument in the first layer(s) for automatic build."
        )

    if not print_fn:
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
