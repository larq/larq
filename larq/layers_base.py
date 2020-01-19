import logging
from typing import Optional

from tensorflow import keras

from larq import metrics as lq_metrics, quantized_scope, quantizers
from larq.quantized_variable import QuantizedVariable
from larq.quantizers import Quantizer

log = logging.getLogger(__name__)


# TODO: find a good way remove duplication between QuantizerBase, QuantizerDepthwiseBase and QuantizerSeparableBase


class BaseLayer(keras.layers.Layer):
    """Base class for defining quantized layers"""

    def get_quantizer(self, name) -> Optional[Quantizer]:
        return None

    def _add_variable_with_custom_getter(self, name: str, **kwargs):
        quantizer = self.get_quantizer(name)
        if quantizer is None:
            return super()._add_variable_with_custom_getter(name, **kwargs)

        old_getter = kwargs.pop("getter")

        # Wrap `getter` with a version that returns a `QuantizedVariable`.
        def getter(*args, **kwargs):
            variable = old_getter(*args, **kwargs)
            return QuantizedVariable.from_variable(variable, quantizer)

        return super()._add_variable_with_custom_getter(name, getter=getter, **kwargs)

    @property
    def non_trainable_weights(self):
        weights = super().non_trainable_weights
        if hasattr(self, "flip_ratio"):
            return [
                weight
                for weight in weights
                if not any(weight is metric_w for metric_w in self.flip_ratio.weights)
            ]
        return weights


class QuantizerBase(BaseLayer):
    """Base class for defining quantized layers

    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Layer`.
    """

    def __init__(
        self, *args, input_quantizer=None, kernel_quantizer=None, metrics=None, **kwargs
    ):
        self.input_quantizer = quantizers.get(input_quantizer)
        self.kernel_quantizer = quantizers.get(kernel_quantizer)
        self._custom_metrics = (
            metrics if metrics is not None else lq_metrics.get_training_metrics()
        )

        super().__init__(*args, **kwargs)
        if kernel_quantizer and not self.kernel_constraint:
            log.warning(
                "Using a weight quantizer without setting `kernel_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def get_quantizer(self, name: str) -> Optional[Quantizer]:
        return self.kernel_quantizer if name == "kernel" else None

    def build(self, input_shape):
        super().build(input_shape)
        if self.kernel_quantizer:
            if "flip_ratio" in self._custom_metrics:
                self.flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.kernel.shape, name=f"flip_ratio/{self.name}"
                )

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        with quantized_scope.scope(True):
            if hasattr(self, "flip_ratio"):
                self.add_metric(self.flip_ratio(self.kernel))
            return super().call(inputs)

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "kernel_quantizer": quantizers.serialize(self.kernel_quantizer),
        }
        return {**super().get_config(), **config}


class QuantizerDepthwiseBase(BaseLayer):
    """Base class for defining quantized layers

    `input_quantizer` and `depthwise_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Layer`.
    """

    def __init__(
        self,
        *args,
        input_quantizer: Optional[Quantizer] = None,
        depthwise_quantizer: Optional[Quantizer] = None,
        metrics=None,
        **kwargs,
    ):
        self.input_quantizer = quantizers.get(input_quantizer)
        self.depthwise_quantizer = quantizers.get(depthwise_quantizer)
        self._custom_metrics = (
            metrics if metrics is not None else lq_metrics.get_training_metrics()
        )

        super().__init__(*args, **kwargs)
        if depthwise_quantizer and not self.depthwise_constraint:
            log.warning(
                "Using a weight quantizer without setting `depthwise_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def get_quantizer(self, name: str) -> Optional[Quantizer]:
        return self.depthwise_quantizer if name == "depthwise_kernel" else None

    def build(self, input_shape):
        super().build(input_shape)
        if self.depthwise_quantizer:
            if "flip_ratio" in self._custom_metrics:
                self.flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.depthwise_kernel.shape,
                    name=f"flip_ratio/{self.name}",
                )

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        with quantized_scope.scope(True):
            if hasattr(self, "flip_ratio"):
                self.add_metric(self.flip_ratio(self.depthwise_kernel))
            return super().call(inputs)

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "depthwise_quantizer": quantizers.serialize(self.depthwise_quantizer),
        }
        return {**super().get_config(), **config}


class QuantizerSeparableBase(BaseLayer):
    """Base class for defining separable quantized layers

    `input_quantizer`, `depthwise_quantizer` and `pointwise_quantizer` are the
    element-wise quantization functions to use. If all quantization functions are `None`
    this layer is equivalent to `SeparableConv1D`. If `use_bias` is True and
    a bias initializer is provided, it adds a bias vector to the output.
    It then optionally applies an activation function to produce the final output.
    """

    def __init__(
        self,
        *args,
        input_quantizer: Optional[Quantizer] = None,
        depthwise_quantizer: Optional[Quantizer] = None,
        pointwise_quantizer: Optional[Quantizer] = None,
        metrics=None,
        **kwargs,
    ):
        self.input_quantizer = quantizers.get(input_quantizer)
        self.depthwise_quantizer = quantizers.get(depthwise_quantizer)
        self.pointwise_quantizer = quantizers.get(pointwise_quantizer)
        self._custom_metrics = (
            metrics if metrics is not None else lq_metrics.get_training_metrics()
        )

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

    def get_quantizer(self, name: str) -> Optional[Quantizer]:
        if name == "depthwise_kernel":
            return self.depthwise_quantizer
        if name == "pointwise_kernel":
            return self.pointwise_quantizer
        return None

    def build(self, input_shape):
        super().build(input_shape)
        if self.depthwise_quantizer:
            if "flip_ratio" in self._custom_metrics:
                self.depthwise_flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.depthwise_kernel.shape,
                    name=f"flip_ratio/{self.name}_depthwise",
                )
        if self.pointwise_quantizer:
            if "flip_ratio" in self._custom_metrics:
                self.pointwise_flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.pointwise_kernel.shape,
                    name=f"flip_ratio/{self.name}_pointwise",
                )

    @property
    def non_trainable_weights(self):
        weights = super().non_trainable_weights
        metrics_weights = []
        if hasattr(self, "depthwise_flip_ratio"):
            metrics_weights.extend(self.depthwise_flip_ratio.weights)
        if hasattr(self, "pointwise_flip_ratio"):
            metrics_weights.extend(self.pointwise_flip_ratio.weights)
        if metrics_weights:
            return [
                weight
                for weight in weights
                if not any(weight is metric_w for metric_w in metrics_weights)
            ]
        return weights

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        with quantized_scope.scope(True):
            if hasattr(self, "depthwise_flip_ratio"):
                self.add_metric(self.depthwise_flip_ratio(self.depthwise_kernel))
            if hasattr(self, "pointwise_flip_ratio"):
                self.add_metric(self.pointwise_flip_ratio(self.pointwise_kernel))
            return super().call(inputs)

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "depthwise_quantizer": quantizers.serialize(self.depthwise_quantizer),
            "pointwise_quantizer": quantizers.serialize(self.pointwise_quantizer),
        }
        return {**super().get_config(), **config}
