# (generated with --quick)

import larq.quantized_variable
from typing import Any, Type

QuantizedVariable: Type[larq.quantized_variable.QuantizedVariable]
log: logging.Logger
logging: module
lq_metrics: module
quantized_scope: module
quantizers: module
tf: module

class BaseLayer(Any):
    __doc__: str
    non_trainable_weights: Any
    def _add_variable_with_custom_getter(self, name, **kwargs) -> Any: ...
    def get_quantizer(self, name) -> None: ...

class QuantizerBase(BaseLayer):
    __doc__: str
    _custom_metrics: Any
    flip_ratio: Any
    input_quantizer: Any
    kernel_quantizer: Any
    def __init__(self, *args, input_quantizer = ..., kernel_quantizer = ..., metrics = ..., **kwargs) -> None: ...
    def build(self, input_shape) -> None: ...
    def call(self, inputs) -> Any: ...
    def get_config(self) -> dict: ...
    def get_quantizer(self, name) -> Any: ...

class QuantizerDepthwiseBase(BaseLayer):
    __doc__: str
    _custom_metrics: Any
    depthwise_quantizer: Any
    flip_ratio: Any
    input_quantizer: Any
    def __init__(self, *args, input_quantizer = ..., depthwise_quantizer = ..., metrics = ..., **kwargs) -> None: ...
    def build(self, input_shape) -> None: ...
    def call(self, inputs) -> Any: ...
    def get_config(self) -> dict: ...
    def get_quantizer(self, name) -> Any: ...

class QuantizerSeparableBase(BaseLayer):
    __doc__: str
    _custom_metrics: Any
    depthwise_flip_ratio: Any
    depthwise_quantizer: Any
    input_quantizer: Any
    non_trainable_weights: Any
    pointwise_flip_ratio: Any
    pointwise_quantizer: Any
    def __init__(self, *args, input_quantizer = ..., depthwise_quantizer = ..., pointwise_quantizer = ..., metrics = ..., **kwargs) -> None: ...
    def build(self, input_shape) -> None: ...
    def call(self, inputs) -> Any: ...
    def get_config(self) -> dict: ...
    def get_quantizer(self, name) -> Any: ...
