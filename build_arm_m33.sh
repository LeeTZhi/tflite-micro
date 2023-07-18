#!/bin/bash

make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR=cmsis_nn microlite -j4
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR=cmsis_nn lib_mfcc -j4
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR=cmsis_nn lib_snore_detection -j4

