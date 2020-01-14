# (generated with --quick)

from typing import Any, Dict

tf: module
utils: module

class WeightClip(Any):
    __doc__: str
    clip_value: Any
    def __call__(self, x) -> Any: ...
    def __init__(self, clip_value = ...) -> None: ...
    def get_config(self) -> Dict[str, Any]: ...

class weight_clip(WeightClip):
    clip_value: Any
