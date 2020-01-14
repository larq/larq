# (generated with --quick)

from typing import Any, Dict, List

__all__: List[str]
approx_sign: Any
dorefa_quantizer: Any
magnitude_aware_sign: Any
math: module
ste_heaviside: Any
ste_sign: Any
ste_tern: Any
swish_sign: Any
tf: module
utils: module

class DoReFaQuantizer(QuantizerFunctionWrapper):
    __doc__: str
    _fn_kwargs: Dict[str, Any]
    fn: Any
    precision: Any
    def __init__(self, k_bit) -> None: ...

class MagnitudeAwareSign(QuantizerFunctionWrapper):
    __doc__: str
    _fn_kwargs: Dict[str, Any]
    fn: Any
    precision: Any
    def __init__(self, clip_value = ...) -> None: ...

class QuantizerFunctionWrapper:
    __doc__: str
    _fn_kwargs: Dict[str, Any]
    precision: Any
    def __call__(self, x) -> Any: ...
    def __init__(self, fn, **kwargs) -> None: ...
    def fn(self) -> Any: ...
    def get_config(self) -> Dict[str, Any]: ...

class SteHeaviside(QuantizerFunctionWrapper):
    __doc__: str
    _fn_kwargs: Dict[str, Any]
    fn: Any
    precision: Any
    def __init__(self, clip_value = ...) -> None: ...

class SteSign(QuantizerFunctionWrapper):
    __doc__: str
    _fn_kwargs: Dict[str, Any]
    fn: Any
    precision: Any
    def __init__(self, clip_value = ...) -> None: ...

class SteTern(QuantizerFunctionWrapper):
    __doc__: str
    _fn_kwargs: Dict[str, Any]
    fn: Any
    precision: Any
    def __init__(self, threshold_value = ..., ternary_weight_networks = ..., clip_value = ...) -> None: ...

class SwishSign(QuantizerFunctionWrapper):
    __doc__: str
    _fn_kwargs: Dict[str, Any]
    fn: Any
    precision: Any
    def __init__(self, beta = ...) -> None: ...

def _clipped_gradient(x, dy, clip_value) -> Any: ...
def _scaled_sign(x) -> Any: ...
def deserialize(name, custom_objects = ...) -> Any: ...
def get(identifier) -> Any: ...
def serialize(quantizer) -> Any: ...
