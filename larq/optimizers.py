import tensorflow as tf
import distutils.version

tf_version = distutils.version.LooseVersion(tf.__version__)
v_1_14 = distutils.version.LooseVersion("1.14.0")

if tf_version >= v_1_14:
    from larq.optimizers_v2 import Bop
else:
    from larq.optimizers_v1 import Bop, XavierLearningRateScaling
