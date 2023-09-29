#!/bin/bash

set -eu

# Path to a flag file
FLAG_FILE="/.container_initialized"
# pip install Cython?
# sudo apt install gdal-bin?
# Check if the flag file exists
if [ ! -f $FLAG_FILE ]; then
    echo "Setting MintPy up for development"
    /root/tools/mambaforge/bin/pip3 install -e /MintPy

    # Create the flag file for future runs
    touch $FLAG_FILE
fi

set +eu

exec "$@"
