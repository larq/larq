from larq import utils

if utils.tf_1_14_or_newer():
    from larq.optimizers_v2 import Bop, BNNOptimizerDuo

    __all__ = ["Bop", "BNNOptimizerDuo"]
else:
    from larq.optimizers_v1 import Bop, XavierLearningRateScaling

    __all__ = ["Bop", "XavierLearningRateScaling"]
