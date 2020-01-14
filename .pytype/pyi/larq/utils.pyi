# (generated with --quick)

from typing import Any, Callable, TypeVar

get_custom_objects: module

_T0 = TypeVar('_T0')

def register_alias(name) -> Callable[[Any], Any]: ...
def register_keras_custom_object(cls: _T0) -> _T0: ...
def set_precision(precision = ...) -> Callable[[Any], Any]: ...
