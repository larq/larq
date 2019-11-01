from larq import utils

if utils.tf_1_14_or_newer():
    from larq.optimizers_v2 import Bop, OptimizerGroup

    __all__ = ["Bop", "OptimizerGroup"]
else:
    from larq.optimizers_v1 import Bop, XavierLearningRateScaling

    __all__ = ["Bop", "XavierLearningRateScaling"]
