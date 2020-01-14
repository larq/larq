# (generated with --quick)

from typing import Any

lq: module
np: module
pytest: module
testing_utils: module
tf: module

class DummyTrainableQuantizer(Any):
    __doc__: str
    dummy_weight: Any
    def build(self, input_shape) -> None: ...
    def call(self, inputs) -> Any: ...

class TestCommonFunctionality:
    __doc__: str
    test_layer_as_quantizer: Any
    test_serialization: Any
    test_serialization_cls: Any
    def test_invalid_usage(self) -> None: ...

class TestGradients:
    __doc__: str
    test_identity_ste_grad: Any
    test_ste_grad: Any
    def test_approx_sign_grad(self, eager_mode) -> None: ...
    def test_dorefa_ste_grad(self, eager_mode) -> None: ...
    def test_magnitude_aware_sign_grad(self, eager_mode) -> None: ...
    def test_swish_grad(self, eager_mode) -> None: ...

class TestQuantization:
    __doc__: str
    test_and_binarization: Any
    test_dorefa_quantize: Any
    test_ternarization_basic: Any
    test_ternarization_with_default_threshold: Any
    test_xnor_binarization: Any
    def test_magnitude_aware_sign_binarization(self, eager_mode) -> None: ...
    def test_ternarization_with_custom_threshold(self) -> None: ...
    def test_ternarization_with_ternary_weight_networks(self) -> None: ...
