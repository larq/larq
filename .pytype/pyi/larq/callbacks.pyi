# (generated with --quick)

from typing import Any, Callable, Optional

keras: module
tf: module

class HyperparameterScheduler(Any):
    __doc__: str
    hyperparameter: str
    optimizer: Any
    schedule: Callable
    verbose: Optional[int]
    def __init__(self, schedule: Callable, hyperparameter: str, optimizer = ..., verbose: Optional[int] = ...) -> None: ...
    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = ...) -> None: ...
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = ...) -> None: ...
    def set_model(self, model) -> None: ...
