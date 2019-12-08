# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_repr[eager] 1'] = "<QuantizedVariable 'x:0' shape=() dtype=float32 quantizer=<lambda> numpy=0.0>"

snapshots['test_repr[eager] 2'] = "<QuantizedVariable 'x:0' shape=() dtype=float32 quantizer=Quantizer numpy=0.0>"

snapshots['test_repr[eager] 3'] = "<QuantizedVariable 'x:0' shape=() dtype=float32 precision=1 numpy=0.0>"

snapshots['test_repr[graph] 1'] = "<QuantizedVariable 'x:0' shape=() dtype=float32 quantizer=<lambda>>"

snapshots['test_repr[graph] 2'] = "<QuantizedVariable 'x:0' shape=() dtype=float32 quantizer=Quantizer>"

snapshots['test_repr[graph] 3'] = "<QuantizedVariable 'x:0' shape=() dtype=float32 precision=1>"
