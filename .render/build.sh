#!/usr/bin/env bash

# Explicitly install and set Python version
pyenv install 3.10.12
pyenv global 3.10.12

# Install pip & setuptools upgrades (safeguard)
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

# Install project dependencies
pip install -r requirements.txt
