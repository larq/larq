# (generated with --quick)

from typing import Any

keras: module
lq: module
lq_testing_utils: module
np: module
pytest: module
testing_utils: module
tf: module

class TestBopOptimizer:
    test_bop_serialization_schedule: Any
    def test_bop_accuracy(self) -> None: ...
    def test_bop_serialization(self) -> None: ...
    def test_bop_tf_1_14_schedules(self) -> None: ...

class TestCaseOptimizer:
    def test_missing_default(self) -> None: ...
    def test_overlapping_predicates(self) -> None: ...
    def test_type_check_default(self) -> None: ...
    def test_type_check_optimizer(self) -> None: ...
    def test_type_check_predicate(self) -> None: ...
    def test_weights(self) -> None: ...

def _assert_weights(weights, expected) -> None: ...
def _test_optimizer(optimizer, target = ..., test_kernels_are_binary = ..., trainable_bn = ...) -> None: ...
def _test_serialization(optimizer) -> None: ...
