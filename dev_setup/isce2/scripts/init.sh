#!/bin/bash

set -eu

# Path to a flag file
FLAG_FILE="/.container_initialized"
# pip install Cython?
# sudo apt install gdal-bin?
# Check if the flag file exists
if [ ! -f $FLAG_FILE ]; then
    echo "BUILDING ISCE FOR DEVELOPMENT"
    pushd /opt/isce2/src/isce2
    mkdir -p build
    cd build
    cmake .. \
        -DPYTHON_MODULE_DIR="$(python3 -c 'import site; print(site.getsitepackages()[-1])')" \
        -DCMAKE_INSTALL_PREFIX=install
    make -j8 install
    popd

    ln -s /usr/lib/python3.8/dist-packages/isce2 /usr/lib/python3.8/dist-packages/isce

    # Create the flag file for future runs
    touch $FLAG_FILE
fi

set +eu
