import logging
from typing import Optional

import tensorflow as tf

from larq import context, quantizers, utils
from larq.quantized_variable import QuantizedVariable
from larq.quantizers import Quantizer

log = logging.getLogger(__name__)


def _compute_padding(stride, dilation_rate, input_size, filter_size):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    output_size = (input_size + stride - 1) // stride
    total_padding = (output_size - 1) * stride + effective_filter_size - input_size
    total_padding = tf.math.maximum(total_padding, 0)
    padding = total_padding // 2
    return padding, padding + (total_padding % 2)


class BaseLayer(tf.keras.layers.Layer):
    """Base class for defining quantized layers.

    `input_quantizer` is the element-wise quantization functions to use.
    If `input_quantizer=None` this layer is equivalent to `tf.keras.layers.Layer`.
    """

    def __init__(self, *args, input_quantizer=None, **kwargs):
        self.input_quantizer = quantizers.get(input_quantizer)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        with context.quantized_scope(True):
            return super().call(inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "input_quantizer": quantizers.serialize(self.input_quantizer),
        }

    def _get_quantizer(self, name) -> Optional[Quantizer]:
        """Get quantizer for given kernel name"""
        return None

    def _add_variable_with_custom_getter(self, name: str, **kwargs):
        quantizer = self._get_quantizer(name)
        if quantizer is None:
            return super()._add_variable_with_custom_getter(name, **kwargs)

        old_getter = kwargs.pop("getter")

        # Wrap `getter` with a version that returns a `QuantizedVariable`.
        def getter(*args, **kwargs):
            variable = old_getter(*args, **kwargs)
            return QuantizedVariable.from_variable(variable, quantizer)

        return super()._add_variable_with_custom_getter(name, getter=getter, **kwargs)


class QuantizerBase(BaseLayer):
    """Base class for defining quantized layers with a single kernel.

    `kernel_quantizer` is the element-wise quantization functions to use.
    If `kernel_quantizer=None` this layer is equivalent to `BaseLayer`.
    """

    def __init__(self, *args, kernel_quantizer=None, **kwargs):
        self.kernel_quantizer = quantizers.get_kernel_quantizer(kernel_quantizer)

        super().__init__(*args, **kwargs)
        if kernel_quantizer and not self.kernel_constraint:
            log.warning(
                "Using a weight quantizer without setting `kernel_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def _get_quantizer(self, name: str) -> Optional[Quantizer]:
        return self.kernel_quantizer if name == "kernel" else None

    def get_config(self):
        return {
            **super().get_config(),
            "kernel_quantizer": quantizers.serialize(self.kernel_quantizer),
        }


class QuantizerBaseConv(tf.keras.layers.Layer):
    """Base class for defining quantized conv layers"""

    def __init__(self, *args, pad_values=0.0, **kwargs):
        self.pad_values = pad_values
        super().__init__(*args, **kwargs)

    def _is_native_padding(self):
        return self.padding != "same" or (
            not tf.is_tensor(self.pad_values) and self.pad_values == 0.0
        )

    def _get_spatial_padding_same(self, shape):
        return [
            _compute_padding(stride, dilation_rate, shape[i], filter_size)
            for i, (stride, dilation_rate, filter_size) in enumerate(
                zip(self.strides, self.dilation_rate, self.kernel_size)
            )
        ]

    def _get_spatial_shape(self, input_shape):
        return (
            input_shape[1:-1]
            if self.data_format == "channels_last"
            else input_shape[2:]
        )

    def _get_padding_same(self, inputs):
        input_shape = tf.shape(inputs)
        padding = self._get_spatial_padding_same(self._get_spatial_shape(input_shape))
        return (
            [[0, 0], *padding, [0, 0]]
            if self.data_format == "channels_last"
            else [[0, 0], [0, 0], *padding]
        )

    def _get_padding_same_shape(self, input_shape):
        spatial_shape = [
            (size + stride - 1) // stride if size is not None else None
            for size, stride in zip(self._get_spatial_shape(input_shape), self.strides)
        ]
        if self.data_format == "channels_last":
            return tf.TensorShape([input_shape[0], *spatial_shape, input_shape[-1]])
        return tf.TensorShape([*input_shape[:2], *spatial_shape])

    def build(self, input_shape):
        if self._is_native_padding():
            super().build(input_shape)
        else:
            with utils.patch_object(self, "padding", "valid"):
                super().build(self._get_padding_same_shape(input_shape))

    def call(self, inputs):
        if self._is_native_padding():
            return super().call(inputs)

        inputs = tf.pad(
            inputs, self._get_padding_same(inputs), constant_values=self.pad_values
        )
        with utils.patch_object(self, "padding", "valid"):
            return super().call(inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "pad_values": tf.keras.backend.get_value(self.pad_values),
        }


class QuantizerDepthwiseBase(BaseLayer):
    """Base class for defining depthwise quantized layers

    `depthwise_quantizer` is the element-wise quantization functions to use.
    If `depthwise_quantizer=None` this layer is equivalent to `BaseLayer`.
    """

    def __init__(
        self, *args, depthwise_quantizer: Optional[Quantizer] = None, **kwargs,
    ):
        self.depthwise_quantizer = quantizers.get_kernel_quantizer(depthwise_quantizer)

        super().__init__(*args, **kwargs)
        if depthwise_quantizer and not self.depthwise_constraint:
            log.warning(
                "Using a weight quantizer without setting `depthwise_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def _get_quantizer(self, name: str) -> Optional[Quantizer]:
        return self.depthwise_quantizer if name == "depthwise_kernel" else None

    def get_config(self):
        return {
            **super().get_config(),
            "depthwise_quantizer": quantizers.serialize(self.depthwise_quantizer),
        }


class QuantizerSeparableBase(BaseLayer):
    """Base class for defining separable quantized layers.

    `depthwise_quantizer` and `pointwise_quantizer` are the element-wise quantization
    functions to use. If all quantization functions are `None` this layer is equivalent
    to `BaseLayer`.
    """

    def __init__(
        self,
        *args,
        depthwise_quantizer: Optional[Quantizer] = None,
        pointwise_quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        self.depthwise_quantizer = quantizers.get_kernel_quantizer(depthwise_quantizer)
        self.pointwise_quantizer = quantizers.get_kernel_quantizer(pointwise_quantizer)

        super().__init__(*args, **kwargs)
        if depthwise_quantizer and not self.depthwise_constraint:
            log.warning(
                "Using `depthwise_quantizer` without setting `depthwise_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )
        if pointwise_quantizer and not self.pointwise_constraint:
            log.warning(
                "Using `pointwise_quantizer` without setting `pointwise_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def _get_quantizer(self, name: str) -> Optional[Quantizer]:
        if name == "depthwise_kernel":
            return self.depthwise_quantizer
        if name == "pointwise_kernel":
            return self.pointwise_quantizer
        return None

    def get_config(self):
        return {
            **super().get_config(),
            "depthwise_quantizer": quantizers.serialize(self.depthwise_quantizer),
            "pointwise_quantizer": quantizers.serialize(self.pointwise_quantizer),
        }
