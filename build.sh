#!/usr/bin/env bash

set -e

python3 -m pip install -r requirements.txt
python3 -m pip install nbconvert git+https://github.com/lgeiger/pydoc-markdown.git
python3 -m pip install -e .[docs]

# Generad API docs
python3 generate_api_docs.py

# run mkdocs
python3 -m mkdocs build
# run mkdocs a second time to make sure generated plots are copied to /site
python3 -m mkdocs build
