# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots[
    "test_summary 1"
] = """\
+sequential stats----------------------------------------------------------------------------------------------------------------+
| Layer                   Input prec.           Outputs  # 1-bit  # 2-bit  # 32-bit  Memory  1-bit MACs  2-bit MACs  32-bit MACs |
|                               (bit)                        x 1      x 1       x 1    (kB)        (kB)        (kB)         (kB) |
+--------------------------------------------------------------------------------------------------------------------------------+
| quant_conv2d                      -  (-1, 64, 64, 32)      288        0        32    0.16           0           0       144.00 |
| max_pooling2d                     -  (-1, 32, 32, 32)        0        0         0       0           0           0            0 |
| quant_depthwise_conv2d            2  (-1, 11, 11, 32)        0      288         0    0.07           0        4.25            0 |
| quant_separable_conv2d            1  (-1, 11, 11, 32)     1312        0        32    0.29       19.38           0            0 |
| flatten                           -        (-1, 3872)        0        0         0       0           0           0            0 |
| dense                             -          (-1, 10)        0        0     38730  151.29           0           0         4.73 |
+--------------------------------------------------------------------------------------------------------------------------------+
| Total                                                     1600      288     38794  151.80       19.38        4.25       148.73 |
+--------------------------------------------------------------------------------------------------------------------------------+
+sequential summary--------------------------+
| Total params                       40682   |
| Trainable params                   40682   |
| Non-trainable params               0       |
| Model size:                        0.15 MB |
| Float-32 Equivalent                0.16 MB |
| Compression Ratio of Memory        0.96    |
| Number of MACs                     1411968 |
| Ratio of MACs that are binarized   0.1124  |
| Ratio of MACs that are ternarized  0.0247  |
+--------------------------------------------+
"""
