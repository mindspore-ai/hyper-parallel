#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "${SCRIPT_DIR}")
NATIVE_ROOT="${PROJECT_ROOT}/build/native"
WORK_ROOT="${NATIVE_ROOT}/work/symmetric_memory"
COMPONENT_ROOT="${NATIVE_ROOT}/components/symmetric_memory/hyper_parallel"
PAYLOAD_COMPONENT_ROOT="${NATIVE_ROOT}/payload/hyper_parallel/core/symmetric_memory"
PAYLOAD_STAGING_ROOT="${NATIVE_ROOT}/payload-staging/symmetric_memory.$$"
FRAMEWORK="all"
SOC_LIST="ascend910b,ascend910_93"
NATIVE_JOBS="$(nproc)"
CLEAN="off"

function show_help() {
    cat <<EOF
Usage:
  bash scripts/build_symmetric_memory.sh [OPTIONS]

Options:
  --framework VALUE   Build framework adapters: mindspore, torch, or all. Default: all.
  --soc-list VALUE    Comma-separated CANN SoC IDs. Default: ascend910b,ascend910_93.
  --jobs VALUE        Parallel build jobs. Default: nproc.
  --clean             Remove symmetric-memory work and install outputs before building.
  -h, --help          Show this help message.
EOF
}

function fail() {
    local reason_code=$1
    local message=$2
    local exit_code=${3:-1}
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${reason_code}"
    echo "ERROR: ${message}" >&2
    exit "${exit_code}"
}

function require_cache_key() {
    local key_name=$1
    local key_value=$2
    if [[ ! "${key_value}" =~ ^[0-9a-f]{16}$ ]]; then
        fail "FRAMEWORK_CACHE_KEY_INVALID" \
            "${key_name} cache identity must be one 16-character hexadecimal digest." 4
    fi
}

function require_value() {
    local option_name=$1
    local option_value=${2:-}
    if [[ -z "${option_value}" || "${option_value}" == --* ]]; then
        fail "INVALID_SELECTION" "${option_name} requires a value." 2
    fi
}

function normalize_framework() {
    local value
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "${value}" in
        mindspore|ms)
            echo "mindspore"
            ;;
        torch|pytorch)
            echo "torch"
            ;;
        all|both)
            echo "all"
            ;;
        *)
            fail "INVALID_SELECTION" \
                "Unsupported --framework '${1}'; use mindspore, torch, or all." 2
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework=*)
            FRAMEWORK=$(normalize_framework "${1#*=}")
            shift
            ;;
        --framework)
            require_value "$1" "${2:-}"
            FRAMEWORK=$(normalize_framework "$2")
            shift 2
            ;;
        --soc-list=*)
            require_value "--soc-list" "${1#*=}"
            SOC_LIST="${1#*=}"
            shift
            ;;
        --soc-list)
            require_value "$1" "${2:-}"
            SOC_LIST="$2"
            shift 2
            ;;
        --jobs=*)
            require_value "--jobs" "${1#*=}"
            NATIVE_JOBS="${1#*=}"
            shift
            ;;
        --jobs)
            require_value "$1" "${2:-}"
            NATIVE_JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN="on"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            fail "INVALID_SELECTION" "Unknown option '$1'." 2
            ;;
    esac
done

if ! [[ "${NATIVE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    fail "INVALID_SELECTION" "--jobs must be a positive integer, got '${NATIVE_JOBS}'." 2
fi

BUILD_SHMEM_TORCH=false
BUILD_SHMEM_MINDSPORE=false
case "${FRAMEWORK}" in
    all)
        BUILD_SHMEM_TORCH=true
        BUILD_SHMEM_MINDSPORE=true
        ;;
    mindspore)
        BUILD_SHMEM_MINDSPORE=true
        ;;
    torch)
        BUILD_SHMEM_TORCH=true
        ;;
esac

CURRENT_REASON_CODE="SYMMETRIC_MEMORY_BUILD_FAILED"
function report_unhandled_error() {
    local exit_code=$?
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${CURRENT_REASON_CODE}"
    echo "ERROR: symmetric memory build failed unexpectedly with exit ${exit_code}." >&2
    exit "${exit_code}"
}
trap report_unhandled_error ERR

cd "${PROJECT_ROOT}"
if [[ "${CLEAN}" == "on" ]]; then
    rm -rf "${WORK_ROOT}" "${COMPONENT_ROOT}"
fi
rm -rf "${COMPONENT_ROOT}" "${PAYLOAD_COMPONENT_ROOT}"
mkdir -p "${COMPONENT_ROOT}"

for required_tool in cmake find gcc g++ grep make python3 readelf readlink sed; do
    if ! command -v "${required_tool}" >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" \
            "Required symmetric-memory build tool not found on PATH: ${required_tool}." 4
    fi
done
PYTHON_CACHE_TAG=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
if [[ "${BUILD_SHMEM_MINDSPORE}" == "true" ]]; then
    if ! command -v ninja >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" "Required MindSpore build tool not found: ninja." 4
    fi
    if ! python3 -c 'import mindspore as ms; assert hasattr(ms.ops, "CustomOpBuilder")' >/dev/null 2>&1; then
        fail "MINDSPORE_CUSTOM_OP_BUILDER_NOT_FOUND" \
            "The active Python environment must provide MindSpore CustomOpBuilder." 4
    fi
fi
if [[ "${BUILD_SHMEM_TORCH}" == "true" ]]; then
    if ! python3 -c 'import torch, torch_npu' >/dev/null 2>&1; then
        fail "TORCH_BUILD_DEPENDENCY_NOT_FOUND" \
            "The active Python environment must provide matching torch and torch_npu packages." 4
    fi
    if ! python3 -c 'import torch; raise SystemExit(0 if torch._C._GLIBCXX_USE_CXX11_ABI else 1)' \
        >/dev/null 2>&1; then
        fail "TORCH_CXX11_ABI_UNSUPPORTED" \
            "The selected Torch native build requires _GLIBCXX_USE_CXX11_ABI=1." 4
    fi
fi

source "${SCRIPT_DIR}/check_gcc_version.sh"
check_gcc_version || fail "UNSUPPORTED_GCC" "Host GCC is outside the supported build range." 5
source "${SCRIPT_DIR}/native/shmem_sdk.sh"
CURRENT_REASON_CODE="SHMEM_SDK_BUILD_FAILED"
hp_prepare_shmem_sdk "${PROJECT_ROOT}" "${SOC_LIST}" "${NATIVE_JOBS}" "${CLEAN}"

SHMEM_INSTALL_ROOT="${HP_SHMEM_INSTALL_ROOT}"
SHMEM_WORK_ROOT="${WORK_ROOT}/toolchain-${HP_SHMEM_TOOLCHAIN_KEY}"
FRAMEWORK_WORK_ROOT="${SHMEM_WORK_ROOT}/framework/${PYTHON_CACHE_TAG}"
SHMEM_SOURCE_DIR="${SHMEM_INSTALL_ROOT}/shmem"
export SHMEM_HOME_PATH="${SHMEM_INSTALL_ROOT}"
export SHMEM_SOURCE_DIR

OPS_BUILD_DIR="${SHMEM_WORK_ROOT}/ops"
OPS_INSTALL_DIR="${COMPONENT_ROOT}/core/symmetric_memory"
CURRENT_REASON_CODE="SYMMETRIC_MEMORY_OPS_BUILD_FAILED"
cmake -S "${PROJECT_ROOT}/hyper_parallel/core/symmetric_memory/ops" \
    -B "${OPS_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OPS_INSTALL_DIR}"
cmake --build "${OPS_BUILD_DIR}" --parallel "${NATIVE_JOBS}"
cmake --install "${OPS_BUILD_DIR}"
SYMMETRIC_MEMORY_KERNEL_LIB="${OPS_INSTALL_DIR}/lib/libaclshmem_symmetric_memory_kernel.so"
if [[ ! -s "${SYMMETRIC_MEMORY_KERNEL_LIB}" ]]; then
    fail "EXPECTED_ARTIFACT_MISSING" \
        "Symmetric-memory kernel library was not installed: ${SYMMETRIC_MEMORY_KERNEL_LIB}." 9
fi

if [[ "${BUILD_SHMEM_TORCH}" == "true" ]]; then
    TORCH_CACHE_KEY=$(python3 -c '
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
torch_spec = find_spec("torch")
npu_spec = find_spec("torch_npu")
torch_version = version("torch")
npu_version = version("torch-npu")
torch_origin = torch_spec.origin if torch_spec else "missing"
npu_origin = npu_spec.origin if npu_spec else "missing"
identity = f"{torch_version}|{torch_origin}|{npu_version}|{npu_origin}"
print(hashlib.sha256(identity.encode()).hexdigest()[:16])
')
    require_cache_key "Torch" "${TORCH_CACHE_KEY}"
    TORCH_BUILD_DIR="${FRAMEWORK_WORK_ROOT}/torch-${TORCH_CACHE_KEY}"
    TORCH_INSTALL_DIR="${COMPONENT_ROOT}/core/symmetric_memory/lib/framework/torch"
    rm -rf "${TORCH_BUILD_DIR}" "${TORCH_INSTALL_DIR}"
    CURRENT_REASON_CODE="TORCH_SYMMETRIC_MEMORY_BUILD_FAILED"
    cmake -S "${PROJECT_ROOT}/hyper_parallel/core/symmetric_memory/platform/torch" \
        -B "${TORCH_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${TORCH_INSTALL_DIR}" \
        -DACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB="${SYMMETRIC_MEMORY_KERNEL_LIB}" \
        -DPython3_EXECUTABLE="$(command -v python3)" \
        -DBUILD_TORCH_LIB=True
    cmake --build "${TORCH_BUILD_DIR}" --parallel "${NATIVE_JOBS}"
    cmake --install "${TORCH_BUILD_DIR}"
fi

if [[ "${BUILD_SHMEM_MINDSPORE}" == "true" ]]; then
    MINDSPORE_CACHE_KEY=$(python3 -c '
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
spec = find_spec("mindspore")
ms_version = version("mindspore")
origin = spec.origin if spec else "missing"
identity = f"{ms_version}|{origin}"
print(hashlib.sha256(identity.encode()).hexdigest()[:16])
')
    require_cache_key "MindSpore" "${MINDSPORE_CACHE_KEY}"
    MINDSPORE_BUILD_DIR="${FRAMEWORK_WORK_ROOT}/mindspore-${MINDSPORE_CACHE_KEY}"
    MINDSPORE_INSTALL_DIR="${COMPONENT_ROOT}/core/symmetric_memory/lib/framework/mindspore"
    rm -rf "${MINDSPORE_BUILD_DIR}" "${MINDSPORE_INSTALL_DIR}"
    CURRENT_REASON_CODE="MINDSPORE_SYMMETRIC_MEMORY_BUILD_FAILED"
    cmake -S "${PROJECT_ROOT}/hyper_parallel/core/symmetric_memory/platform/mindspore" \
        -B "${MINDSPORE_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${MINDSPORE_INSTALL_DIR}" \
        -DACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB="${SYMMETRIC_MEMORY_KERNEL_LIB}" \
        -DPython3_EXECUTABLE="$(command -v python3)" \
        -DBUILD_MS_LIB=True
    cmake --build "${MINDSPORE_BUILD_DIR}" --parallel "${NATIVE_JOBS}"
    if [[ ! -s "${MINDSPORE_BUILD_DIR}/kernel_meta/aclshmem_ms.so" ]]; then
        fail "EXPECTED_ARTIFACT_MISSING" \
            "MindSpore CustomOpBuilder returned without producing aclshmem_ms.so." 9
    fi
    cmake --install "${MINDSPORE_BUILD_DIR}"
fi

CURRENT_REASON_CODE="SYMMETRIC_MEMORY_ELF_VALIDATION_FAILED"
for private_library in \
    libhyper_parallel_shmem.so \
    libhyper_parallel_shmem_utils.so \
    aclshmem_bootstrap_uid.so \
    aclshmem_bootstrap_config_store.so; do
    private_library_path="${OPS_INSTALL_DIR}/lib/shmem/${private_library}"
    if [[ ! -s "${private_library_path}" ]]; then
        fail "EXPECTED_ARTIFACT_MISSING" "Private SHMEM library is missing: ${private_library_path}." 10
    fi
    if ! readelf -d "${private_library_path}" | grep -Eq "\\(SONAME\\).*\\[${private_library}\\]"; then
        fail "SHMEM_PRIVATE_SONAME_INVALID" \
            "Private SHMEM SONAME does not match ${private_library}: ${private_library_path}." 10
    fi
done

while IFS= read -r component_library; do
    dynamic_section=$(readelf -d "${component_library}")
    if grep -Eq '\(NEEDED\).*\[(libshmem|libshmem_utils|libshmem_bootstrap_[^]]*)\.so\]' \
        <<< "${dynamic_section}"; then
        fail "GENERIC_SHMEM_DT_NEEDED_FOUND" \
            "Component library depends on a generic SHMEM SONAME: ${component_library}." 10
    fi
    while IFS= read -r runpath_entry; do
        runpath_value=$(sed -n 's/.*\[\(.*\)\].*/\1/p' <<< "${runpath_entry}")
        IFS=':' read -r -a search_paths <<< "${runpath_value}"
        for search_path in "${search_paths[@]}"; do
            if [[ "${search_path}" == /* ]]; then
                fail "ABSOLUTE_RUNPATH_FOUND" \
                    "Component library contains an absolute RPATH/RUNPATH: ${component_library}: ${search_path}." 10
            fi
        done
    done < <(grep -E '(RPATH|RUNPATH)' <<< "${dynamic_section}" || true)
done < <(find "${OPS_INSTALL_DIR}" -type f -name '*.so' -print)

rm -rf "${PAYLOAD_STAGING_ROOT}"
mkdir -p "$(dirname "${PAYLOAD_STAGING_ROOT}")" "$(dirname "${PAYLOAD_COMPONENT_ROOT}")"
cp -a "${COMPONENT_ROOT}/core/symmetric_memory" "${PAYLOAD_STAGING_ROOT}"
mv "${PAYLOAD_STAGING_ROOT}" "${PAYLOAD_COMPONENT_ROOT}"

trap - ERR
echo "INFO: symmetric memory build completed"
echo "  framework: ${FRAMEWORK}"
echo "  component: ${COMPONENT_ROOT}"
echo "  PYTHONPATH payload: ${PAYLOAD_COMPONENT_ROOT}"
