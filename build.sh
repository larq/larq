#!/usr/bin/env bash

set -e

pip install -r requirements.txt
pip install nbconvert git+https://github.com/lgeiger/pydoc-markdown.git
pip install -e .[docs]

# Generad API docs
python generate_api_docs.py

# run mkdocs
mkdocs build
