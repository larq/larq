from larq.utils import get_tf_version_major_minor_float

if get_tf_version_major_minor_float() > 1.13:
    from larq.optimizers_v2 import *
else:
    from larq.optimizers_v1 import *
