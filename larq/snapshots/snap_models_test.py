# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_summary 1'] = '''+sequential stats-----------------------------------------------------------+
| Layer                    Outputs  # 1-bit  # 2-bit  # 32-bit  Memory (kB) |
+---------------------------------------------------------------------------+
| quant_conv2d    (-1, 26, 26, 32)      288        0        32         0.16 |
| max_pooling2d   (-1, 13, 13, 32)        0        0         0         0.00 |
| quant_conv2d_1  (-1, 11, 11, 32)        0     9216        32         2.38 |
| dense           (-1, 11, 11, 10)        0        0       330         1.29 |
+---------------------------------------------------------------------------+
| Total                                 288     9216       394         3.82 |
+---------------------------------------------------------------------------+
+sequential summary--------------+
| Total params           9898    |
| Trainable params       9898    |
| Non-trainable params   0       |
| Float-32 Equivalent    0.04 MB |
| Compression of Memory  10.11   |
+--------------------------------+
'''
