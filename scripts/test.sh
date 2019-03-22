#!/bin/sh

set -ev  # print commands as they're executed and immediately fail if one has a non zero exit code

pip install tensorflow==$TF_VERSION pytest

pytest .
