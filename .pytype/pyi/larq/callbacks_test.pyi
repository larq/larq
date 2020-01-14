# (generated with --quick)

import larq.callbacks
from typing import Any, List, Type

HyperparameterScheduler: Type[larq.callbacks.HyperparameterScheduler]
lq: module
lq_testing_utils: module
np: module
pytest: module
testing_utils: module
tf: module

class LogHistory(Any):
    batches: List[nothing]
    epochs: List[nothing]
    def _store_logs(self, storage, batch_or_epoch, logs = ...) -> None: ...
    def on_batch_end(self, batch, logs = ...) -> None: ...
    def on_epoch_end(self, epoch, logs = ...) -> None: ...
    def on_train_begin(self, logs = ...) -> None: ...

class TestHyperparameterScheduler:
    def test_case_optimizer(self) -> None: ...
    def test_no_optimizer(self) -> None: ...
    def test_normal_optimizer(self) -> None: ...
