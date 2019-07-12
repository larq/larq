#!/usr/bin/env bash

set -e
basedir=`dirname $0`

# Clean generated output
rm -r $basedir/docs/models || true
rm -r $basedir/docs/api || true

mkdir $basedir/docs/models

# Generad API docs
python $basedir/generate_api_docs.py

# Fetch larq-zoo docs
curl -o $basedir/docs/models/index.md https://raw.githubusercontent.com/larq/zoo/master/docs/index.md
curl -o $basedir/docs/models/examples.md https://raw.githubusercontent.com/larq/zoo/master/docs/examples.md
curl -o $basedir/docs/models/api.md https://raw.githubusercontent.com/larq/zoo/master/docs/api.md

# run mkdocs
mkdocs $1
