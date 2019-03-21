#!/bin/bash

set -v  # print commands as they're executed

pip install tensorflow==$TF_VERSION pytest

pytest .
