import logging
import tensorflow as tf
from larq import quantizers, utils, metrics as lq_metrics

log = logging.getLogger(__name__)


# TODO: find a good way remove duplication between QuantizerBase, QuantizerDepthwiseBase and QuantizerSeparableBase


class QuantizerBase(tf.keras.layers.Layer):
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
        self.quantized_latent_weights = []
        self.quantizers = []
        self._custom_metrics = (
            metrics if metrics is not None else lq_metrics.get_training_metrics()
        )

        super().__init__(*args, **kwargs)
        if kernel_quantizer and not self.kernel_constraint:
            log.warning(
                "Using a weight quantizer without setting `kernel_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def build(self, input_shape):
        super().build(input_shape)
        if self.kernel_quantizer:
            self.quantized_latent_weights.append(self.kernel)
            self.quantizers.append(self.kernel_quantizer)
            if "flip_ratio" in self._custom_metrics and utils.supports_metrics():
                self.flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.kernel.shape, name=f"flip_ratio/{self.name}"
                )

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

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)

        with utils.quantize(self, "kernel", self.kernel_quantizer) as kernel:
            if hasattr(self, "flip_ratio"):
                self.add_metric(self.flip_ratio(kernel))
            return super().call(inputs)

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "kernel_quantizer": quantizers.serialize(self.kernel_quantizer),
        }
        return {**super().get_config(), **config}


class QuantizerDepthwiseBase(tf.keras.layers.Layer):
    """Base class for defining quantized layers

    `input_quantizer` and `depthwise_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Layer`.
    """

    def __init__(
        self,
        *args,
        input_quantizer=None,
        depthwise_quantizer=None,
        metrics=None,
        **kwargs,
    ):
        self.input_quantizer = quantizers.get(input_quantizer)
        self.depthwise_quantizer = quantizers.get(depthwise_quantizer)
        self.quantized_latent_weights = []
        self.quantizers = []
        self._custom_metrics = (
            metrics if metrics is not None else lq_metrics.get_training_metrics()
        )

        super().__init__(*args, **kwargs)
        if depthwise_quantizer and not self.depthwise_constraint:
            log.warning(
                "Using a weight quantizer without setting `depthwise_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def build(self, input_shape):
        super().build(input_shape)
        if self.depthwise_quantizer:
            self.quantized_latent_weights.append(self.depthwise_kernel)
            self.quantizers.append(self.depthwise_quantizer)
            if "flip_ratio" in self._custom_metrics and utils.supports_metrics():
                self.flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.depthwise_kernel.shape,
                    name=f"flip_ratio/{self.name}",
                )

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

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)

        with utils.quantize(
            self, "depthwise_kernel", self.depthwise_quantizer
        ) as kernel:
            if hasattr(self, "flip_ratio"):
                self.add_metric(self.flip_ratio(kernel))
            return super().call(inputs)

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "depthwise_quantizer": quantizers.serialize(self.depthwise_quantizer),
        }
        return {**super().get_config(), **config}


class QuantizerSeparableBase(tf.keras.layers.Layer):
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
        input_quantizer=None,
        depthwise_quantizer=None,
        pointwise_quantizer=None,
        metrics=None,
        **kwargs,
    ):
        self.input_quantizer = quantizers.get(input_quantizer)
        self.depthwise_quantizer = quantizers.get(depthwise_quantizer)
        self.pointwise_quantizer = quantizers.get(pointwise_quantizer)
        self.quantized_latent_weights = []
        self.quantizers = []
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

    def build(self, input_shape):
        super().build(input_shape)
        if self.depthwise_quantizer:
            self.quantized_latent_weights.append(self.depthwise_kernel)
            self.quantizers.append(self.depthwise_quantizer)
            if "flip_ratio" in self._custom_metrics and utils.supports_metrics():
                self.depthwise_flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.depthwise_kernel.shape,
                    name=f"flip_ratio/{self.name}_depthwise",
                )
        if self.pointwise_quantizer:
            self.quantized_latent_weights.append(self.pointwise_kernel)
            self.quantizers.append(self.pointwise_quantizer)
            if "flip_ratio" in self._custom_metrics and utils.supports_metrics():
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

        with utils.quantize(
            self, "depthwise_kernel", self.depthwise_quantizer
        ) as depthwise_kernel, utils.quantize(
            self, "pointwise_kernel", self.pointwise_quantizer
        ) as pointwise_kernel:
            if hasattr(self, "depthwise_flip_ratio"):
                self.add_metric(self.depthwise_flip_ratio(depthwise_kernel))
            if hasattr(self, "pointwise_flip_ratio"):
                self.add_metric(self.pointwise_flip_ratio(pointwise_kernel))
            return super().call(inputs)

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "depthwise_quantizer": quantizers.serialize(self.depthwise_quantizer),
            "pointwise_quantizer": quantizers.serialize(self.pointwise_quantizer),
        }
        return {**super().get_config(), **config}
