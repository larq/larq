from importlib import metadata

from larq import (  # pytype: disable=pyi-error
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
