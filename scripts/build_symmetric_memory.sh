#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
SET_ENV_FILE="/usr/local/Ascend/cann/set_env.sh"
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "Warning: ASCEND_HOME_PATH is not set. Attempting to source ${SET_ENV_FILE} to set it."
    source "${SET_ENV_FILE}"
    if [ -z "${ASCEND_HOME_PATH}" ]; then
        echo "Error: After sourcing ${SET_ENV_FILE}, ASCEND_HOME_PATH is still not set."
        exit 1
    fi
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
BUILD_DIR=$PROJECT_ROOT/build
export HP_THIRD_PARTY_DIR="${PROJECT_ROOT}/3rdparty"
mkdir -p $HP_THIRD_PARTY_DIR

BUILD_TYPE=RELEASE

USE_CXX11_ABI=ON

COMPILE_OPTIONS=""

SHMEM_VERSION=v1.3.0

cd ${PROJECT_ROOT}

PREBUILD_DIR="./prebuild"
TAR_FILE="${PREBUILD_DIR}/symmetric_memory.tar.gz"
SHA256_FILE="${PREBUILD_DIR}/symmetric_memory.tar.gz.sha256"
DEST_DIR="./build/lib/hyper_parallel"

PREBUILD_SHMEM="${HYPER_PREBUILD_SHMEM:-true}"

# 目标 GCC 版本（可通过环境变量 TARGET_GCC_VERSION 覆盖，当bisheng编译器的头文件版本<7.3.0时会寻找/usr/local下面的对应版本）
TARGET_GCC_VERSION=${TARGET_GCC_VERSION:-"7.3.0"}
# 手动指定 GCC 根目录（优先级最高，适配特殊环境）
MANUAL_GCC_ROOT=${MANUAL_GCC_ROOT:-""}


function is_bisheng_header_le_730() {
    bisheng -v -E -x c++ 2>&1 /dev/null | \
    grep -E "^ /.*/include/c\+\+/[0-9]+(\.[0-9]+)*(/|$)" | head -1 | \
    sed -nE 's/.*c\+\+\/([0-9]+(\.[0-9]+)*).*/\1/p' | \
    awk -F '.' '{
        v = $1 * 10000 + ($2 ? $2 * 100 : 0) + ($3 ? $3 : 0); exit(v < 70300 ? 0 : 1);
}'
}

function detect_gcc_path() {
    local target_version=$1
    local arch=$(uname -m)  # 获取系统架构（x86_64/aarch64 等）
    echo "===== 开始探测 ${arch} 架构下的 GCC ${target_version} 路径 ====="

    # 步骤 1：优先使用手动指定的路径
    if [ -n "${MANUAL_GCC_ROOT}" ] && [ -d "${MANUAL_GCC_ROOT}" ]; then
        echo "✅ 检测到手动指定的 GCC 路径：${MANUAL_GCC_ROOT}"
        local gcc_bin="${MANUAL_GCC_ROOT}/bin/gcc"
        local gxx_bin="${MANUAL_GCC_ROOT}/bin/g++"
        # 验证版本
        if ${gcc_bin} --version 2>/dev/null | grep -q "${target_version}"; then
            export GCC_ROOT="${MANUAL_GCC_ROOT}"
            export GCC_BIN="${gcc_bin}"
            export GXX_BIN="${gxx_bin}"
            return 0
        else
            echo "❌ 手动指定的 GCC 路径版本不匹配！"
        fi
    fi

    # 步骤 2：自动探测常见的 GCC 安装路径（按优先级排序）
    local gcc_paths=(
        "/usr/local/gcc/gcc${target_version//./}"  # /usr/local/gcc/gcc730
        "/usr/local/gcc-${target_version}"         # /usr/local/gcc-7.3.0
        "/usr/local"                               # 通用本地路径
        "/usr"                                     # 系统路径
    )

    for path in "${gcc_paths[@]}"; do
        local gcc_bin="${path}/bin/gcc"
        local gxx_bin="${path}/bin/g++"
        if [ -x "${gcc_bin}" ] && ${gcc_bin} --version 2>/dev/null | grep -q "${target_version}"; then
            echo "✅ 自动探测到 GCC ${target_version} 路径：${path}"
            export GCC_ROOT="${path}"
            export GCC_BIN="${gcc_bin}"
            export GXX_BIN="${gxx_bin}"
            return 0
        fi
    done

    # 步骤 3：探测失败，提示用户
    echo "❌ 未找到 GCC ${target_version}！请检查安装或通过 MANUAL_GCC_ROOT 指定路径"
    echo "    示例：export MANUAL_GCC_ROOT=/usr/local/gcc/gcc730 && bash build.sh"
    return 1
}

function adapt_gcc_headers_libs() {
    local arch=$(uname -m)
    local gcc_version=$(echo ${TARGET_GCC_VERSION} | cut -d '.' -f1-2)  # 7.3

    # 步骤 1：自动识别头文件路径（适配不同架构）
    export GCC_HEADER_ROOT=$(
        find "${GCC_ROOT}" -path "*/include/c++/${TARGET_GCC_VERSION}" -type d 2>/dev/null | head -1
    )
    if [ -z "${GCC_HEADER_ROOT}" ]; then
        # 兼容短版本号（7.3 → 7.3.0）
        GCC_HEADER_ROOT=$(find "${GCC_ROOT}" -path "*/include/c++/${gcc_version}" -type d 2>/dev/null | head -1)
    fi

    # 步骤 2：自动识别架构专属头文件路径
    export GCC_ARCH_HEADER=$(
        find "${GCC_HEADER_ROOT}" -path "*/${arch}*linux*" -type d 2>/dev/null | head -1
    )
    if [ -z "${GCC_ARCH_HEADER}" ]; then
        GCC_ARCH_HEADER="${GCC_HEADER_ROOT}/${arch}-unknown-linux-gnu"
        echo "⚠️ 未找到架构专属头文件路径，使用默认：${GCC_ARCH_HEADER}"
    fi

    # 步骤 3：自动识别库路径
    export GCC_LIB_PATH=$(
        find "${GCC_ROOT}" -path "*/lib64" -name "libstdc++.so*" -type f -print0 2>/dev/null | xargs -0 dirname | head -1
    )
    if [ -z "${GCC_LIB_PATH}" ]; then
        GCC_LIB_PATH="${GCC_ROOT}/lib64"
    fi

    export GCC_VERSION_PATH=$(
        find "${GCC_ROOT}" -path "*/lib*/gcc/${arch}*-linux-gnu/${TARGET_GCC_VERSION}" -type d 2>/dev/null | head -1
    )
    # 验证路径有效性
    echo "===== GCC 路径适配结果 ====="
    echo "GCC 根目录：${GCC_ROOT}"
    echo "C++ 头文件根目录：${GCC_HEADER_ROOT}"
    echo "架构专属头文件：${GCC_ARCH_HEADER}"
    echo "库文件路径：${GCC_LIB_PATH}"
    echo "GCC 版本路径：${GCC_VERSION_PATH}"

    if [ ! -d "${GCC_HEADER_ROOT}" ] || [ ! -d "${GCC_LIB_PATH}" ]; then
        echo "❌ GCC 头文件/库路径无效！"
        return 1
    fi

    # 步骤 4：导出环境变量，供 bisheng 使用
    export CPLUS_INCLUDE_PATH="${GCC_ARCH_HEADER}:${GCC_HEADER_ROOT}:${CPLUS_INCLUDE_PATH}"
    export LIBRARY_PATH="${GCC_LIB_PATH}:${GCC_VERSION_PATH}:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${GCC_LIB_PATH}:${LD_LIBRARY_PATH}"
    # 编译参数：强制 bisheng 使用该 GCC 路径
    export BISHENG_CXX_FLAGS="-nostdinc++ \
    -isystem ${GCC_ARCH_HEADER} \
    -isystem ${GCC_HEADER_ROOT} \
    --gcc-toolchain=${GCC_ROOT} \
    -fno-lto\
    -fPIC -mllvm --disable-symbolication=1"
    export LD_FLAGS="-L${GCC_LIB_PATH} -L${GCC_VERSION_PATH} -Wl,-rpath=${GCC_LIB_PATH} -Wl"
    return 0
}

function set_bisheng_env() {
    # 步骤 1：探测 GCC 路径
    if ! detect_gcc_path "${TARGET_GCC_VERSION}"; then
        exit 1
    fi

    # 步骤 2：适配头文件/库路径
    if ! adapt_gcc_headers_libs; then
        exit 1
    fi

    # 步骤 3：验证 bisheng 是否使用正确的 GCC 路径
    echo "===== 验证 bisheng 关联的 GCC 头文件 ====="
    # 定义目标版本为最低依赖版本
    TARGET_GCC_VERSION="7.3.0"

    # 成功输出 success，失败输出 fail + 实际头文件路径
    bisheng ${BISHENG_CXX_FLAGS} -v -E -x c++ 2>&1 /dev/null | \
    awk -v target="${TARGET_GCC_VERSION}" '
    /include.*c\+\+\/[0-9.]+/ {
        path = $0; 
        # 提取路径中的版本号（兼容 7.3.0/7.3/4.8.5 格式）
        if (match(path, /c\+\+\/([0-9]+\.[0-9]+\.?[0-9]*)/, ver)) {
            header_ver = ver[1];
            # 版本号数值化（用于对比）
            split(header_ver, v, ".");
            header_num = v[1]*10000 + v[2]*100 + (v[3]? v[3] : 0);
            split(target, t, ".");
            target_num = t[1]*10000 + t[2]*100 + (t[3]? t[3] : 0);
            # 判断是否匹配目标版本
            if (header_num >= target_num) {
                matched = 1;
            } else {
                fail_path = path;  # 记录失败时的实际路径
            }
        }
    }
    # 处理结束后输出结果
    END {
        if (matched) {
            print "success" path;
        } else {
            print "fail, actual C++ header path: " fail_path;
        }
    }'

}


function build_shmem() {
    echo "Start to build shmem."
    cd $HP_THIRD_PARTY_DIR; [[ ! -d "shmem" ]] && git clone --depth 1 https://gitcode.com/cann/shmem.git -b $SHMEM_VERSION; cd $PROJECT_ROOT
    if [ ! -d "3rdparty/shmem" ];
    then
        echo "shmem does not exists."
        exit 1
    fi

    cd 3rdparty/shmem
    PATCH_FILE="../patch/shmem.patch"

    # 第一步：检查 Patch 文件是否存在
    if [ ! -f "$PATCH_FILE" ]; then
        echo "Error: Patch file $PATCH_FILE not exists！"
        exit 1
    fi

    if ! grep -q "hpshmem" src/CMakeLists.txt; then
        git apply "$PATCH_FILE"
    fi
    echo "shmem build environment prepared, start to build shmem."
    export CACHE_DIR=$PWD/../
    rm -rf install
    bash scripts/build.sh
    source install/set_env.sh
    cd -
}

function build_aclshmem_ops() {
    echo "build aclshmem ops start"
    cd "$PROJECT_ROOT"/hyper_parallel/core/symmetric_memory/ops || exit
    OPS_BUILD_DIR=$BUILD_DIR/symmetric_memory/ops
    rm -rf $OPS_BUILD_DIR
    mkdir -p $OPS_BUILD_DIR
    OPS_INSTALL_DIR="$BUILD_DIR"/lib/hyper_parallel
    cmake -S . --no-warn-unused-cli -B $OPS_BUILD_DIR $COMPILE_OPTIONS -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DUSE_CXX11_ABI=$USE_CXX11_ABI -DCMAKE_INSTALL_PREFIX=$OPS_INSTALL_DIR
    
    cmake --build $OPS_BUILD_DIR -j
    cmake --install $OPS_BUILD_DIR
    cd "$PROJECT_ROOT" || exit
    echo "build aclshmem ops success"
}

function build_torch_library() {
    echo "build torch library start"
    cd "$PROJECT_ROOT"/hyper_parallel/core/symmetric_memory/platform/torch || exit
    TORCH_BUILD_DIR=$BUILD_DIR/symmetric_memory/torch
    rm -rf $TORCH_BUILD_DIR
    mkdir -p $TORCH_BUILD_DIR
    TORCH_INSTALL_DIR="$BUILD_DIR"/lib/hyper_parallel/platform/torch/symmetric_memory
    cmake -S ./ --no-warn-unused-cli -B $TORCH_BUILD_DIR $COMPILE_OPTIONS -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DUSE_CXX11_ABI=$USE_CXX11_ABI -DCMAKE_INSTALL_PREFIX=$TORCH_INSTALL_DIR -DPython3_EXECUTABLE="$(which python3)" -DBUILD_TORCH_LIB=True
    cmake --build $TORCH_BUILD_DIR -j
    cmake --install $TORCH_BUILD_DIR
    cd "$PROJECT_ROOT" || exit
    echo "build torch library success"
}

function build_ms_library() {
    echo "build mindspore library start"
    cd "$PROJECT_ROOT"/hyper_parallel/core/symmetric_memory/platform/mindspore || exit
    MS_BUILD_DIR="$BUILD_DIR"/symmetric_memory/mindspore
    rm -rf $MS_BUILD_DIR
    mkdir -p $MS_BUILD_DIR
    MS_INSTALL_DIR="$BUILD_DIR"/lib/hyper_parallel/platform/mindspore/symmetric_memory
    cmake -S ./ --no-warn-unused-cli -B $MS_BUILD_DIR $COMPILE_OPTIONS -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DUSE_CXX11_ABI=$USE_CXX11_ABI -DCMAKE_INSTALL_PREFIX=$MS_INSTALL_DIR -DPython3_EXECUTABLE="$(which python3)" -DBUILD_MS_LIB=True
    cmake --build $MS_BUILD_DIR -j
    cmake --install $MS_BUILD_DIR
    cd "$PROJECT_ROOT" || exit
    echo "build mindspore library success"
}

set -e

echo "============================================="
echo "Prebuild mode: ${PREBUILD_SHMEM} (true=use prebuilt package, false=compile from source)"
echo "============================================="

# 2. If prebuild mode is enabled
if [[ "${PREBUILD_SHMEM}" == "true" ]]; then
    echo -e "\n[1/4] Checking for prebuilt files..."
    # Check if tarball and hash file exist
    if [[ ! -f "${TAR_FILE}" || ! -f "${SHA256_FILE}" ]]; then
        echo "ERROR: Prebuilt files missing, switching to compile mode"
        PREBUILD_SHMEM="false"
    else
        echo "[2/4] Verifying SHA256 checksum..."
        # Enter directory to avoid path issues
        cd "${PREBUILD_DIR}" || exit 1

        # Read expected hash (trim whitespace/newlines)
        EXPECTED_SHA=$(cat "$(basename "${SHA256_FILE}")" | tr -d ' \n\r')
        # Calculate actual file hash
        ACTUAL_SHA=$(sha256sum "$(basename "${TAR_FILE}")" | awk '{print $1}')

        # Return to original directory
        cd - > /dev/null || exit 1

        echo "Expected hash: ${EXPECTED_SHA}"
        echo "Actual hash:   ${ACTUAL_SHA}"

        if [[ "${EXPECTED_SHA}" == "${ACTUAL_SHA}" ]]; then
            echo -e "\n[3/4] SHA256 check passed! Extracting files..."
            # Create target directory
            mkdir -p "${DEST_DIR}"
            # Extract to destination
            tar -zxf "${TAR_FILE}" -C "${DEST_DIR}"
            echo "[4/4] Prebuilt package extracted successfully! Path: ${DEST_DIR}"
            echo -e "\n✅ Prebuild process completed, exiting script"
        else
            echo -e "\n❌ Hash mismatch, switching to compile mode"
            PREBUILD_SHMEM="false"
        fi
    fi
fi

if is_bisheng_header_le_730; then
    echo "Detected bisheng header version <= 7.3.0, applying compatibility adjustments."
    set_bisheng_env
fi
if [[ "$PREBUILD_SHMEM" != "true" ]]; then
    build_shmem
    build_aclshmem_ops
else
    cd $HP_THIRD_PARTY_DIR; [[ ! -d "shmem" ]] && git clone --depth 1 https://gitcode.com/cann/shmem.git -b $SHMEM_VERSION; cd $PROJECT_ROOT
    if [ ! -d "3rdparty/shmem" ];
    then
        echo "shmem does not exists."
        exit 1
    fi
fi
build_torch_library
build_ms_library

cd ${CURRENT_DIR}
