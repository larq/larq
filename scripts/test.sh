#!/bin/bash

set -v  # print commands as they're executed

pip install tensorflow==$1 pytest

pytest .
