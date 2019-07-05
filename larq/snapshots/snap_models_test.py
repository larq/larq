# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_summary 1'] = '''+sequential stats-------------------------------------------------------+
| Layer                         Outputs  # 1-bit  # 32-bit  Memory (kB) |
+-----------------------------------------------------------------------+
| quant_conv2d         (-1, 26, 26, 32)      288        32         0.16 |
| max_pooling2d        (-1, 13, 13, 32)        0         0         0.00 |
| batch_normalization  (-1, 13, 13, 32)        0       128         0.50 |
+-----------------------------------------------------------------------+
| Total                                      288       160         0.66 |
+-----------------------------------------------------------------------+
+sequential summary--------------+
| Total params           448     |
| Trainable params       384     |
| Non-trainable params   64      |
| Float-32 Equivalent    0.00 MB |
| Compression of Memory  2.65    |
+--------------------------------+

'''
