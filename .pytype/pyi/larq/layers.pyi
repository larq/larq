# (generated with --quick)

import larq.layers_base
from typing import Any, Type

QuantizerBase: Type[larq.layers_base.QuantizerBase]
QuantizerDepthwiseBase: Type[larq.layers_base.QuantizerDepthwiseBase]
QuantizerSeparableBase: Type[larq.layers_base.QuantizerSeparableBase]
tf: module
utils: module

class QuantConv1D(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., data_format = ..., dilation_rate = ..., activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantConv2D(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., data_format = ..., dilation_rate = ..., activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantConv2DTranspose(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., output_padding = ..., data_format = ..., dilation_rate = ..., activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantConv3D(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., data_format = ..., dilation_rate = ..., activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantConv3DTranspose(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., output_padding = ..., data_format = ..., activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantDense(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, units, activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantDepthwiseConv2D(larq.layers_base.QuantizerDepthwiseBase, Any):
    __doc__: str
    def __init__(self, kernel_size, strides = ..., padding = ..., depth_multiplier = ..., data_format = ..., activation = ..., use_bias = ..., input_quantizer = ..., depthwise_quantizer = ..., depthwise_initializer = ..., bias_initializer = ..., depthwise_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., depthwise_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantLocallyConnected1D(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., data_format = ..., activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., implementation = ..., **kwargs) -> None: ...

class QuantLocallyConnected2D(larq.layers_base.QuantizerBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., data_format = ..., activation = ..., use_bias = ..., input_quantizer = ..., kernel_quantizer = ..., kernel_initializer = ..., bias_initializer = ..., kernel_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., kernel_constraint = ..., bias_constraint = ..., metrics = ..., implementation = ..., **kwargs) -> None: ...

class QuantSeparableConv1D(larq.layers_base.QuantizerSeparableBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., data_format = ..., dilation_rate = ..., depth_multiplier = ..., activation = ..., use_bias = ..., input_quantizer = ..., depthwise_quantizer = ..., pointwise_quantizer = ..., depthwise_initializer = ..., pointwise_initializer = ..., bias_initializer = ..., depthwise_regularizer = ..., pointwise_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., depthwise_constraint = ..., pointwise_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...

class QuantSeparableConv2D(larq.layers_base.QuantizerSeparableBase, Any):
    __doc__: str
    def __init__(self, filters, kernel_size, strides = ..., padding = ..., data_format = ..., dilation_rate = ..., depth_multiplier = ..., activation = ..., use_bias = ..., input_quantizer = ..., depthwise_quantizer = ..., pointwise_quantizer = ..., depthwise_initializer = ..., pointwise_initializer = ..., bias_initializer = ..., depthwise_regularizer = ..., pointwise_regularizer = ..., bias_regularizer = ..., activity_regularizer = ..., depthwise_constraint = ..., pointwise_constraint = ..., bias_constraint = ..., metrics = ..., **kwargs) -> None: ...
