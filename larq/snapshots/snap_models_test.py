# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_summary 1'] = '''+sequential stats----------------------------------------------------------------------------------------------------------------+
| Layer                   Input prec.           Outputs  # 1-bit  # 2-bit  # 32-bit  Memory  1-bit MACs  2-bit MACs  32-bit MACs |
|                               (bit)                        x 1      x 1       x 1    (kB)                                      |
+--------------------------------------------------------------------------------------------------------------------------------+
| quant_conv2d                      -  (-1, 64, 64, 32)      288        0        32    0.16           0           0      1179648 |
| max_pooling2d                     -  (-1, 32, 32, 32)        0        0         0       0           0           0            0 |
| quant_depthwise_conv2d            2  (-1, 11, 11, 32)        0      288         0    0.07           0       34848            0 |
| batch_normalization               -  (-1, 11, 11, 32)        0        0        64    0.25           0           0            0 |
| quant_separable_conv2d            1  (-1, 11, 11, 32)     1312        0        32    0.29      158752           0            0 |
| flatten                           -        (-1, 3872)        0        0         0       0           0           0            0 |
| dense                             -          (-1, 10)        0        0     38730  151.29           0           0        38720 |
+--------------------------------------------------------------------------------------------------------------------------------+
| Total                                                     1600      288     38858  152.05      158752       34848      1218368 |
+--------------------------------------------------------------------------------------------------------------------------------+
+sequential summary-----------------------------+
| Total params                       40.7 k     |
| Trainable params                   1.98 k     |
| Non-trainable params               38.8 k     |
| Model size                         152.05 KiB |
| Model size (8-bit FP weights)      38.21 KiB  |
| Float-32 Equivalent                159.16 KiB |
| Compression Ratio of Memory        0.96       |
| Number of MACs                     1.41 M     |
| Ratio of MACs that are binarized   0.1124     |
| Ratio of MACs that are ternarized  0.0247     |
+-----------------------------------------------+
'''

snapshots['test_summary 2'] = '''+sequential_1 stats--------------------+
| Layer   Input prec.  Outputs  Memory |
|               (bit)             (kB) |
+--------------------------------------+
| lambda            -     (2,)       0 |
+--------------------------------------+
| Total                              0 |
+--------------------------------------+
+sequential_1 summary-------------------+
| Total params                   0      |
| Trainable params               0      |
| Non-trainable params           0      |
| Model size                     0.00 B |
| Model size (8-bit FP weights)  0.00 B |
| Float-32 Equivalent            0.00 B |
| Compression Ratio of Memory    0.00   |
| Number of MACs                 0      |
+---------------------------------------+
'''
