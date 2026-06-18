#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build custom ops MindSpore extension: hyper_parallel_custom_ops_ms.so
#
# This script compiles the .cc files under
#   hyper_parallel/platform/mindspore/custom_ops/
# into a pre-built .so using MindSpore's CustomOpBuilder.build().
# The output is placed under build/lib/ relative to the custom_ops source dir
# and is picked up by setup.py's package_data at install time.
#
# BUILD_CUSTOM_OPS_EXTENSION controls whether this component is built:
#   unset / off: skip custom ops
#   on: build custom ops

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
CUSTOM_OPS_SRC="$PROJECT_ROOT/hyper_parallel/platform/mindspore/custom_ops"
CUSTOM_OPS_BUILD_DIR="$CUSTOM_OPS_SRC/build"
BUILD_CUSTOM_OPS_EXTENSION="${BUILD_CUSTOM_OPS_EXTENSION:-}"
BUILD_CUSTOM_OPS_EXTENSION_DISPLAY="${BUILD_CUSTOM_OPS_EXTENSION:-unset}"

case "${BUILD_CUSTOM_OPS_EXTENSION}" in
    on|ON)
        ;;
    ""|off|OFF)
        rm -rf "$CUSTOM_OPS_BUILD_DIR"
        echo "INFO: custom ops build skipped (BUILD_CUSTOM_OPS_EXTENSION=${BUILD_CUSTOM_OPS_EXTENSION_DISPLAY})."
        exit 0
        ;;
    *)
        echo "ERROR: Unsupported BUILD_CUSTOM_OPS_EXTENSION=${BUILD_CUSTOM_OPS_EXTENSION}."
        echo "       Use unset/off to skip custom ops, or on to build."
        exit 1
        ;;
esac

SET_ENV_FILE="/usr/local/Ascend/cann/set_env.sh"
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "Warning: ASCEND_HOME_PATH is not set. Attempting to source ${SET_ENV_FILE}."
    if [ -f "${SET_ENV_FILE}" ]; then
        source "${SET_ENV_FILE}"
    fi
    if [ -z "${ASCEND_HOME_PATH}" ]; then
        echo "Error: ASCEND_HOME_PATH is still not set. Please set it manually."
        exit 1
    fi
fi

# Host GCC must be in [7.3.0, 11.3.0] (mindspore-aligned).
source "${SCRIPT_DIR}/check_gcc_version.sh"
check_gcc_version || exit 1

BUILD_TYPE="${CMAKE_BUILD_TYPE:-RELEASE}"

echo "============================================="
echo "Building custom_ops MindSpore library"
echo "  Source : $CUSTOM_OPS_SRC"
echo "  Build  : $CUSTOM_OPS_BUILD_DIR"
echo "  Type   : $BUILD_TYPE"
echo "============================================="

if [ ! -d "$CUSTOM_OPS_SRC" ]; then
    echo "Error: Source directory not found: $CUSTOM_OPS_SRC"
    exit 1
fi

cd "$CUSTOM_OPS_SRC" || exit 1
rm -rf "$CUSTOM_OPS_BUILD_DIR"
mkdir -p "$CUSTOM_OPS_BUILD_DIR"

cmake -S . \
    --no-warn-unused-cli \
    -B "$CUSTOM_OPS_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DPython3_EXECUTABLE="$(which python3)"

cmake --build "$CUSTOM_OPS_BUILD_DIR" -j

cd "$PROJECT_ROOT" || exit 1

echo "============================================="
echo "✓ build custom_ops mindspore library success"
echo "  Output: $CUSTOM_OPS_BUILD_DIR/lib/"
echo "============================================="
