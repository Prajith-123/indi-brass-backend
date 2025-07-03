#!/bin/bash

# Fix missing system packages for pip builds
apt-get update
apt-get install -y python3-distutils

# Install Python packages locally for Vercel
pip install --upgrade -r requirements.txt --target .
