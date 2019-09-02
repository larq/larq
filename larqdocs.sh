#!/usr/bin/env bash

set -e
basedir=`dirname $0`

# Generad API docs
python $basedir/generate_api_docs.py

# run mkdocs
mkdocs $1
