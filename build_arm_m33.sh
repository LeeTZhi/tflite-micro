#!/bin/bash
BUILD_TYPE=release
gmake -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=$BUILD_TYPE microlite  -j4
gmake -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=$BUILD_TYPE lib_mfcc  -j4
gmake -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=$BUILD_TYPE lib_snore_detection -j4

