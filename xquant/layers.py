import tensorflow as tf
from xquant import utils


class QuantizerBase(tf.keras.layers.Layer):
    def __init__(self, *args, kernel_quantizer=None, input_quantizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_quantizer = kernel_quantizer
        self.input_quantizer = input_quantizer

    def call(self, inputs):
        if self.kernel_quantizer:
            self.kernel = self.kernel_quantizer(self.kernel)
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        return super().call(inputs)


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
