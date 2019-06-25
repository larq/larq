import tensorflow as tf
from distutils.version import LooseVersion

if LooseVersion(tf.__version__) >= LooseVersion("1.14.0"):
    from larq.optimizers_v2 import Bop

    __all__ = ["Bop"]
else:
    from larq.optimizers_v1 import Bop, XavierLearningRateScaling

    __all__ = ["Bop", "XavierLearningRateScaling"]
