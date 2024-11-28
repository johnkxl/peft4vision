#!/bin/bash
DF_ANALYZE_PATH=$1
VENV=".peft_venv"
ACTIVATE="$VENV/bin/activate"
PYTHON="$VENV/bin/python"

VERSION="3.12.5"
pyenv install "$VERSION" --skip-existing
pyenv shell $VERSION

# Create Virtual Environment.
python -m venv .peft_venv --clear
# Activate virtual environment.
source "$ACTIVATE"

# Install dependencies.
pip install -r requirements.txt
pip install -e "$DF_ANALYZE_PATH"