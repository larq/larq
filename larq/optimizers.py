from larq import utils

if utils.tf_1_14_or_newer():
    from larq.optimizers_v2 import *
else:
    from larq.optimizers_v1 import *
