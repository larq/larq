import tensorflow as tf


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


class QuantConv1D(QuantizerBase, tf.keras.layers.Conv1D):
    pass


class QuantConv2D(QuantizerBase, tf.keras.layers.Conv2D):
    pass


class QuantConv3D(QuantizerBase, tf.keras.layers.Conv3D):
    pass


class QuantConv2DTranspose(QuantizerBase, tf.keras.layers.Conv2DTranspose):
    pass


class QuantConv3DTranspose(QuantizerBase, tf.keras.layers.Conv3DTranspose):
    pass


class QuantLocallyConnected1D(QuantizerBase, tf.keras.layers.LocallyConnected1D):
    pass


class QuantLocallyConnected2D(QuantizerBase, tf.keras.layers.LocallyConnected2D):
    pass


class QuantDense(QuantizerBase, tf.keras.layers.Dense):
    pass
