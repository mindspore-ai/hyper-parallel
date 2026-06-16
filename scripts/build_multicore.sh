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
#   - MindSpore custom op (hyper_parallel_mega_moe_ms) — enabled by default
#   - PyTorch NpuExtension (hyper_parallel_mega_moe_pta) — disabled by default
#
# BUILD_MULTICORE_EXTENSION controls multicore build scope:
#   unset / 0: skip multicore entirely
#   1: build MindSpore multicore
#   2: build PyTorch multicore
#   all: build both MindSpore multicore and PyTorch multicore

# ── User-configurable flags ────────────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
BUILD_MULTICORE_EXTENSION="${BUILD_MULTICORE_EXTENSION:-}"
BUILD_MULTICORE_EXTENSION_DISPLAY="${BUILD_MULTICORE_EXTENSION:-unset}"
BUILD_MULTICORE_MINDSPORE=false
BUILD_MULTICORE_TORCH=false

case "${BUILD_MULTICORE_EXTENSION}" in
    all|ALL|both|BOTH)
        BUILD_MULTICORE_MINDSPORE=true
        BUILD_MULTICORE_TORCH=true
        ;;
    ""|0|skip|SKIP|none|NONE|false|FALSE)
        ;;
    1|mindspore|MINDSPORE|ms|MS)
        BUILD_MULTICORE_MINDSPORE=true
        ;;
    2|torch|TORCH|pytorch|PYTORCH)
        BUILD_MULTICORE_TORCH=true
        ;;
    *)
        echo "ERROR: Unsupported BUILD_MULTICORE_EXTENSION=${BUILD_MULTICORE_EXTENSION}."
        echo "       Use unset/0 to skip, 1 for MindSpore, 2 for PyTorch, or all for both."
        exit 1
        ;;
esac
# ──────────────────────────────────────────────────────────────────────────────

set -e

clean_skipped_multicore_outputs() {
    if [ "${BUILD_MULTICORE_MINDSPORE}" != "true" ]; then
        rm -rf "$PROJECT_ROOT/hyper_parallel/core/multicore/platform/mindspore/build"
    fi
    if [ "${BUILD_MULTICORE_TORCH}" != "true" ]; then
        rm -f "$PROJECT_ROOT"/hyper_parallel/core/multicore/platform/torch/*.so
    fi
}

if [ "${BUILD_MULTICORE_MINDSPORE}" != "true" ] && [ "${BUILD_MULTICORE_TORCH}" != "true" ]; then
    clean_skipped_multicore_outputs
    echo "INFO: multicore build skipped (BUILD_MULTICORE_EXTENSION=${BUILD_MULTICORE_EXTENSION_DISPLAY})."
    exit 0
fi

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

# ── Extract prebuild vendors tarball ──────────────────────────────────────────
PREBUILD_DIR="$PROJECT_ROOT/hyper_parallel/core/multicore/prebuild"
TARBALL="$PREBUILD_DIR/mega_moe.tar.gz"
if [ -f "$TARBALL" ] && [ ! -d "$PREBUILD_DIR/mega_moe" ]; then
    echo "Extracting prebuild vendors: $TARBALL"
    tar -xzf "$TARBALL" -C "$PREBUILD_DIR"
fi

BUILD_TYPE="${CMAKE_BUILD_TYPE:-RELEASE}"

echo "============================================="
echo "BUILD_MULTICORE_EXTENSION: ${BUILD_MULTICORE_EXTENSION_DISPLAY}"
echo "Build MindSpore multicore: ${BUILD_MULTICORE_MINDSPORE}"
echo "Build PyTorch multicore: ${BUILD_MULTICORE_TORCH}"
echo "============================================="

clean_skipped_multicore_outputs

# ── Build MindSpore extension ──────────────────────────────────────────────────
if [ "${BUILD_MULTICORE_MINDSPORE}" = "true" ]; then
    MULTICORE_MS_SRC="$PROJECT_ROOT/hyper_parallel/core/multicore/platform/mindspore"
    MULTICORE_MS_BUILD_DIR="$MULTICORE_MS_SRC/build"

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
    VENDORS_SRC="$PREBUILD_DIR/mega_moe/vendors"
    VENDORS_DST="$PROJECT_ROOT/build/lib/hyper_parallel/core/multicore/prebuild/mega_moe"
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
else
    echo "INFO: MindSpore multicore build skipped (BUILD_MULTICORE_EXTENSION=${BUILD_MULTICORE_EXTENSION_DISPLAY})."
fi

# ── Build PyTorch extension (opt-in) ──────────────────────────────────────────
if [ "${BUILD_MULTICORE_TORCH}" = "true" ]; then
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
    echo "INFO: PyTorch multicore build skipped (BUILD_MULTICORE_EXTENSION=${BUILD_MULTICORE_EXTENSION_DISPLAY})."
fi
