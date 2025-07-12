#!/bin/bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MW_PATH="${MW_PATH:-$CURRENT_DIR/3rd/middleware/v2}"
TPU_PATH="${TPU_SDK_INSTALL_PATH:-$CURRENT_DIR/3rd/tpu}"
IVE_PATH="${IVE_SDK_INSTALL_PATH:-$CURRENT_DIR/3rd/ive}"
KERNEL_ROOT="${KERNEL_ROOT:-$CURRENT_DIR}"
USE_TPU_IVE="${USE_TPU_IVE:-OFF}"
CHIP_ARCH="${CHIP_ARCH:-CV181X}"
SDK_VER="${SDK_VER:-musl_riscv64}"

CLEAN=$1

echo "enter compile_sample.sh !!!"
echo "MW_PATH: $MW_PATH"
echo "TPU_PATH: $TPU_PATH"
echo "IVE_PATH: $IVE_PATH"
echo "KERNEL_ROOT: $KERNEL_ROOT"
echo "USE_TPU_IVE: $USE_TPU_IVE"
echo "CHIP_ARCH: $CHIP_ARCH"
echo "SDK_VER: $SDK_VER"

MAKE_OPTS=("KERNEL_ROOT=$KERNEL_ROOT" "MW_PATH=$MW_PATH" "TPU_PATH=$TPU_PATH"
           "IVE_PATH=$IVE_PATH" "USE_TPU_IVE=$USE_TPU_IVE"
           "CHIP=$CHIP_ARCH" "SDK_VER=$SDK_VER" "-j10")

build_and_clean() {
    pushd $1 || exit 1

    if [ "$CLEAN" = "clean" ]; then
        make clean || exit 1
    else
        make "${MAKE_OPTS[@]}" || exit 1
    fi

    echo "$1 done"
    popd
}

build_and_clean "${CURRENT_DIR}/cvi_yolov1"

echo "finish build sample!!!"

