#!/bin/bash

BUILD_TYPE=release
TOOLS_PATH="/usr/local/Cellar/gcc/12.2.0/bin"
COMMON_FLAGS="-f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=$BUILD_TYPE"
COMMON_TOOLCHAIN="TOOLCHAIN=$TOOLS_PATH/gcc-12 CC_TOOL=$TOOLS_PATH/gcc-12 CXX_TOOL=$TOOLS_PATH/g++-12 AR_TOOL=$TOOLS_PATH/gcc-ar-12"

#if is macosx set CC & CXX
if [[ "$OSTYPE" == "darwin"* ]]; then
    export CC=$TOOLS_PATH/gcc-12
    export CXX=$TOOLS_PATH/g++-12
fi

gmake $COMMON_FLAGS clean

gmake $COMMON_FLAGS $COMMON_TOOLCHAIN microlite -j4
gmake $COMMON_FLAGS $COMMON_TOOLCHAIN lib_mfcc -j4
gmake $COMMON_FLAGS $COMMON_TOOLCHAIN lib_snore_detection -j4
