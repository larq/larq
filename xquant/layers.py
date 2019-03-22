import tensorflow as tf
from xquant import utils
from xquant import quantizers


class QuantizerBase(tf.keras.layers.Layer):
    def __init__(self, *args, input_quantizer=None, kernel_quantizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer = quantizers.get(input_quantizer)
        self.kernel_quantizer = quantizers.get(kernel_quantizer)

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        if self.kernel_quantizer:
            full_precision_kernel = self.kernel
            self.kernel = self.kernel_quantizer(self.kernel)

        output = super().call(inputs)
        if self.kernel_quantizer:
            # Reset the full precision kernel to make keras eager tests pass.
            # Is this a problem with our unit tests or a real bug?
            self.kernel = full_precision_kernel
        return output

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "kernel_quantizer": quantizers.serialize(self.kernel_quantizer),
        }
        return {**super().get_config(), **config}


class QuantizerSeparableBase(tf.keras.layers.Layer):
    def __init__(
        self,
        *args,
        input_quantizer=None,
        depthwise_quantizer=None,
        pointwise_quantizer=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_quantizer = quantizers.get(input_quantizer)
        self.depthwise_quantizer = quantizers.get(depthwise_quantizer)
        self.pointwise_quantizer = quantizers.get(pointwise_quantizer)

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        if self.depthwise_quantizer:
            full_precision_depthwise_kernel = self.depthwise_kernel
            self.depthwise_kernel = self.depthwise_quantizer(self.depthwise_kernel)
        if self.pointwise_quantizer:
            full_precision_pointwise_kernel = self.pointwise_kernel
            self.pointwise_kernel = self.pointwise_quantizer(self.pointwise_kernel)

        output = super().call(inputs)
        # Reset the full precision kernel to make keras eager tests pass.
        # Is this a problem with our unit tests or a real bug?
        if self.depthwise_quantizer:
            self.depthwise_kernel = full_precision_depthwise_kernel
        if self.pointwise_quantizer:
            self.pointwise_kernel = full_precision_pointwise_kernel
        return output

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "depthwise_quantizer": quantizers.serialize(self.depthwise_quantizer),
            "pointwise_quantizer": quantizers.serialize(self.pointwise_quantizer),
        }
        return {**super().get_config(), **config}


@utils.register_keras_custom_object
class QuantConv1D(QuantizerBase, tf.keras.layers.Conv1D):
    pass


@utils.register_keras_custom_object
class QuantConv2D(QuantizerBase, tf.keras.layers.Conv2D):
    pass


@utils.register_keras_custom_object
class QuantConv3D(QuantizerBase, tf.keras.layers.Conv3D):
    pass


@utils.register_keras_custom_object
class QuantConv2DTranspose(QuantizerBase, tf.keras.layers.Conv2DTranspose):
    pass


@utils.register_keras_custom_object
class QuantConv3DTranspose(QuantizerBase, tf.keras.layers.Conv3DTranspose):
    pass


@utils.register_keras_custom_object
class QuantLocallyConnected1D(QuantizerBase, tf.keras.layers.LocallyConnected1D):
    pass


@utils.register_keras_custom_object
class QuantLocallyConnected2D(QuantizerBase, tf.keras.layers.LocallyConnected2D):
    pass


@utils.register_keras_custom_object
class QuantDense(QuantizerBase, tf.keras.layers.Dense):
    pass


@utils.register_keras_custom_object
class QuantSeparableConv1D(QuantizerSeparableBase, tf.keras.layers.SeparableConv1D):
    pass


@utils.register_keras_custom_object
class QuantSeparableConv2D(QuantizerSeparableBase, tf.keras.layers.SeparableConv2D):
    pass
