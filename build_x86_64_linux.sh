#!/bin/bash
BUILD_TYPE := release
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=$BUILD_TYPE microlite  -j4
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=$BUILD_TYPE lib_mfcc  -j4
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=$BUILD_TYPE lib_snore_detection -j4

