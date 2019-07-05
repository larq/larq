# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_summary 1'] = '''+sequential stats-------------------------------------------------+
| Layer                   Outputs  # 1-bit  # 32-bit  Memory (kB) |
+-----------------------------------------------------------------+
| quant_conv2d   (-1, 26, 26, 32)      288        32         0.16 |
| max_pooling2d  (-1, 13, 13, 32)        0         0         0.00 |
| dense          (-1, 13, 13, 10)        0       330         1.29 |
+-----------------------------------------------------------------+
| Total                                288       362         1.45 |
+-----------------------------------------------------------------+
+sequential summary--------------+
| Total params           650     |
| Trainable params       650     |
| Non-trainable params   0       |
| Float-32 Equivalent    0.00 MB |
| Compression of Memory  1.75    |
+--------------------------------+
'''
