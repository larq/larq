from larq import (
    activations,
    callbacks,
    constraints,
    context,
    layers,
    math,
    metrics,
    models,
    optimizers,
    quantizers,
    utils,
)

try:
    from importlib import metadata  # type: ignore
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata  # type: ignore

__version__ = metadata.version("larq")

__all__ = [
    "layers",
    "activations",
    "callbacks",
    "constraints",
    "context",
    "math",
    "metrics",
    "models",
    "quantizers",
    "optimizers",
    "utils",
]
