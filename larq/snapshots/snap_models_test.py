# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_summary 1'] = '''+sequential stats---------------------------------------------------------------------------+
| Layer                   Input prec.           Outputs  # 1-bit  # 2-bit  # 32-bit  Memory |
|                               (bit)                                                  (kB) |
+-------------------------------------------------------------------------------------------+
| quant_conv2d                      -  (-1, 64, 64, 32)      288        0        32    0.16 |
| max_pooling2d                     -  (-1, 32, 32, 32)        0        0         0    0.00 |
| quant_depthwise_conv2d            2  (-1, 11, 11, 32)        0    32768        32    8.12 |
| quant_separable_conv2d            1  (-1, 11, 11, 32)     1312        0        32    0.29 |
| dense                             -  (-1, 11, 11, 10)        0        0       330    1.29 |
+-------------------------------------------------------------------------------------------+
| Total                                                     1600    32768       426    9.86 |
+-------------------------------------------------------------------------------------------+
+sequential summary--------------+
| Total params           34794   |
| Trainable params       34794   |
| Non-trainable params   0       |
| Float-32 Equivalent    0.13 MB |
| Compression of Memory  13.79   |
+--------------------------------+
'''
