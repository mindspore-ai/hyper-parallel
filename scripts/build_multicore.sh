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
# Build multicore op extensions:
#   - MindSpore custom op (hyper_parallel_multicore_moe_ffn_ms) — always built
#   - PyTorch NpuExtension (hyper_parallel_multicore_moe_ffn_pta) — opt-in, set below
#
# To enable the PyTorch extension build, change the variable below to "true":
#   BUILD_TORCH_EXTENSION=true

# ── User-configurable flags ────────────────────────────────────────────────────
# Set to "true" to compile the PyTorch (torch_npu) extension.
# Requires: torch, torch_npu, and CANN vendor libs (CANN_VENDOR_FWD_LIBDIR /
#           CANN_VENDOR_BWD_LIBDIR or CANN_VENDOR_LIBDIR) to be available.
BUILD_TORCH_EXTENSION=false
# ──────────────────────────────────────────────────────────────────────────────

set -e

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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# ── Extract prebuild vendors tarball ──────────────────────────────────────────
PREBUILD_DIR="$PROJECT_ROOT/hyper_parallel/core/multicore/prebuild"
TARBALL="$PREBUILD_DIR/multicore_moe_ffn.tar.gz"
if [ -f "$TARBALL" ] && [ ! -d "$PREBUILD_DIR/multicore_moe_ffn" ]; then
    echo "Extracting prebuild vendors: $TARBALL"
    tar -xzf "$TARBALL" -C "$PREBUILD_DIR"
fi

# ── Build MindSpore extension ──────────────────────────────────────────────────
MULTICORE_MS_SRC="$PROJECT_ROOT/hyper_parallel/core/multicore/platform/mindspore"
MULTICORE_MS_BUILD_DIR="$MULTICORE_MS_SRC/build"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-RELEASE}"

echo "============================================="
echo "Building multicore MindSpore library"
echo "  Source : $MULTICORE_MS_SRC"
echo "  Build  : $MULTICORE_MS_BUILD_DIR"
echo "  Type   : $BUILD_TYPE"
echo "============================================="

if [ ! -d "$MULTICORE_MS_SRC" ]; then
    echo "Error: Source directory not found: $MULTICORE_MS_SRC"
    exit 1
fi

cd "$MULTICORE_MS_SRC" || exit 1
rm -rf "$MULTICORE_MS_BUILD_DIR"
mkdir -p "$MULTICORE_MS_BUILD_DIR"

cmake -S . \
    --no-warn-unused-cli \
    -B "$MULTICORE_MS_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DPython3_EXECUTABLE="$(which python3)"

cmake --build "$MULTICORE_MS_BUILD_DIR" -j

# ── Copy prebuild vendor libs into the build output dir ───────────────────────
# Place vendor packages alongside the compiled extension so that all multicore
# runtime assets are co-located under build/lib/ and picked up by package_data.
VENDORS_SRC="$PREBUILD_DIR/multicore_moe_ffn/vendors"
VENDORS_DST="$PROJECT_ROOT/build/lib/hyper_parallel/core/multicore/prebuild/multicore_moe_ffn"
if [ -d "$VENDORS_SRC" ]; then
    rm -rf "$VENDORS_DST"
fi
mkdir -p "$VENDORS_DST"
echo "Copying prebuild vendors to build output: $VENDORS_DST"
cp -r "$VENDORS_SRC" "$VENDORS_DST"

cd "$PROJECT_ROOT" || exit 1

echo "============================================="
echo "✓ build multicore mindspore library success"
echo "  Output: $MULTICORE_MS_BUILD_DIR/lib/"
echo "============================================="

# ── Build PyTorch extension (opt-in) ──────────────────────────────────────────
if [ "${BUILD_TORCH_EXTENSION}" = "true" ]; then
    MULTICORE_TORCH_SRC="$PROJECT_ROOT/hyper_parallel/core/multicore/platform/torch"

    echo "============================================="
    echo "Building multicore PyTorch extension"
    echo "  Source : $MULTICORE_TORCH_SRC"
    echo "============================================="

    if [ ! -d "$MULTICORE_TORCH_SRC" ]; then
        echo "Error: Source directory not found: $MULTICORE_TORCH_SRC"
        exit 1
    fi

    cd "$MULTICORE_TORCH_SRC" || exit 1

    # build_ext --inplace places the compiled .so next to setup.py so that
    # platform/torch/__init__.py can import it directly as a sibling module.
    python3 setup.py build_ext --inplace

    cd "$PROJECT_ROOT" || exit 1

    echo "============================================="
    echo "✓ build multicore pytorch extension success"
    echo "  Output: $MULTICORE_TORCH_SRC/"
    echo "============================================="
else
    echo "INFO: PyTorch extension build skipped (BUILD_TORCH_EXTENSION=false)."
    echo "      To enable, set BUILD_TORCH_EXTENSION=true in $0"
fi
