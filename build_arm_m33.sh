#!/bin/bash
BUILD_TYPE=release
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOOLS_PATH="/usr/local/Cellar/gcc/12.2.0/bin"
    Make=gmake
else
    Make=make
fi

$Make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=$BUILD_TYPE microlite  -j4
$Make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=$BUILD_TYPE lib_mfcc  -j4
$Make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=$BUILD_TYPE lib_snore_detection -j4

