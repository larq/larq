# (generated with --quick)

import contextlib
from typing import Any, Callable, Iterator, List, Set, TypeVar

FlipRatio: Any
_AVAILABLE_METRICS: Set[str]
_GLOBAL_TRAINING_METRICS: set
__all__: List[str]
np: module
scope: Callable[..., contextlib._GeneratorContextManager]
tf: module
utils: module

_T = TypeVar('_T')

def contextmanager(func: Callable[..., Iterator[_T]]) -> Callable[..., contextlib._GeneratorContextManager[_T]]: ...
def get_training_metrics() -> set: ...
