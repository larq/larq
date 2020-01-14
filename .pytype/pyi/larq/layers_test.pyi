# (generated with --quick)

from typing import Any, List, Tuple

PARAMS_ALL_LAYERS: List[Tuple[Any, Any, Tuple[int, ...], dict]]
PARAMS_SEP_LAYERS: List[Tuple[Any, Any, Tuple[int, ...]]]
inspect: module
lq: module
np: module
pytest: module
test_layer_kwargs: Any
testing_utils: module
tf: module

class TestLayerWarns:
    def test_depthwise_layer_does_not_warn(self, caplog) -> None: ...
    def test_depthwise_layer_warns(self, caplog) -> None: ...
    def test_layer_does_not_warn(self, caplog) -> None: ...
    def test_layer_warns(self, caplog) -> None: ...
    def test_separable_layer_does_not_warn(self, caplog) -> None: ...
    def test_separable_layer_warns(self, caplog) -> None: ...

class TestLayers:
    test_binarization: Any
    test_separable_layers: Any
    def test_depthwise_layers(self, keras_should_run_eagerly) -> None: ...

def test_metrics() -> None: ...
