#!/bin/bash

ROOT_PWD=$(cd "$(dirname $0)" && cd -P "$(dirname "$SOURCE")" && pwd)

# Clean up previous builds
if [ "$1" = "clean" ]; then
    if [ -d "${ROOT_PWD}/build" ]; then
        rm -rf "${ROOT_PWD}/build"
        echo " ${ROOT_PWD}/build has been deleted!"
    fi

    if [ -d "${ROOT_PWD}/install" ]; then
        rm -rf "${ROOT_PWD}/install"
        echo " ${ROOT_PWD}/install has been deleted!"
    fi
    exit
fi

EXAMPLE_NAME="luckfox_pico_yolov1"
EXAMPLE_DIR="./${EXAMPLE_NAME}"

if [[ -d "$EXAMPLE_DIR" ]]; then
    if [ -d "${ROOT_PWD}/build" ]; then
        rm -rf "${ROOT_PWD}/build"
    fi
    mkdir "${ROOT_PWD}/build"
    cd "${ROOT_PWD}/build"
    
    cmake .. -DEXAMPLE_DIR="$EXAMPLE_DIR" -DEXAMPLE_NAME="$EXAMPLE_NAME"
    make -j install
else
    echo "Error: Directory $EXAMPLE_DIR does not exist!"
fi
