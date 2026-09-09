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
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
NATIVE_ROOT="${PROJECT_ROOT}/build/native"
WORK_ROOT="${NATIVE_ROOT}/work/multicore/shmem"
COMPONENT_ROOT="${NATIVE_ROOT}/components/multicore/hyper_parallel"
SOC_LIST="ascend910b,ascend910_93"
NATIVE_JOBS="$(nproc)"
CLEAN="off"

function show_help() {
    cat <<EOF
Usage:
  bash hyper_parallel/core/multicore/shmem/build.sh [OPTIONS]

Options:
  --soc-list VALUE    Comma-separated CANN SoC IDs. Default: ascend910b,ascend910_93.
  --jobs VALUE        Parallel build jobs. Default: nproc.
  --clean             Remove SHMEM work and install outputs before building.
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


while [[ $# -gt 0 ]]; do
    case "$1" in
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


CURRENT_REASON_CODE="SHMEM_BUILD_FAILED"
PYTHON_BIN=python
function report_unhandled_error() {
    local exit_code=$?
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${CURRENT_REASON_CODE}"
    echo "ERROR: SHMEM build failed unexpectedly with exit ${exit_code}." >&2
    exit "${exit_code}"
}
trap report_unhandled_error ERR

cd "${PROJECT_ROOT}"
if [[ "${CLEAN}" == "on" ]]; then
    rm -rf "${WORK_ROOT}" "${COMPONENT_ROOT}/core/multicore/shmem"
fi
rm -rf "${COMPONENT_ROOT}/core/multicore/shmem"
mkdir -p "${COMPONENT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "PYTHON_BUILD_DEPENDENCY_NOT_FOUND" \
        "The selected Python executable is unavailable: ${PYTHON_BIN}." 4
fi
PYTHON_BIN=$(command -v "${PYTHON_BIN}")
for required_tool in cmake find gcc g++ grep make readelf readlink sed; do
    if ! command -v "${required_tool}" >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" \
            "Required SHMEM build tool not found on PATH: ${required_tool}." 4
    fi
done
PYTHON_CACHE_TAG=$("${PYTHON_BIN}" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
if ! TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON_BIN}" -c '
from importlib.util import find_spec
import torch
raise SystemExit(0 if find_spec("torch_npu") is not None else 1)
' >/dev/null 2>&1; then
    fail "TORCH_BUILD_DEPENDENCY_NOT_FOUND" \
        "The active Python environment must provide matching torch and torch_npu packages." 4
fi

source "${PROJECT_ROOT}/scripts/check_gcc_version.sh"
check_gcc_version || fail "UNSUPPORTED_GCC" "Host GCC is outside the supported build range." 5
source "${PROJECT_ROOT}/hyper_parallel/core/multicore/shmem/_build/shmem_sdk.sh"
CURRENT_REASON_CODE="SHMEM_SDK_BUILD_FAILED"
hp_prepare_shmem_sdk "${PROJECT_ROOT}" "${SOC_LIST}" "${NATIVE_JOBS}" "${CLEAN}"

SHMEM_INSTALL_ROOT="${HP_SHMEM_INSTALL_ROOT}"
SHMEM_WORK_ROOT="${WORK_ROOT}/toolchain-${HP_SHMEM_TOOLCHAIN_KEY}"
FRAMEWORK_WORK_ROOT="${SHMEM_WORK_ROOT}/framework/${PYTHON_CACHE_TAG}"
SHMEM_SOURCE_DIR="${SHMEM_INSTALL_ROOT}/shmem"
export SHMEM_HOME_PATH="${SHMEM_INSTALL_ROOT}"
export SHMEM_SOURCE_DIR

OPS_INSTALL_DIR="${COMPONENT_ROOT}/core/multicore/shmem"

TORCH_CACHE_KEY=$("${PYTHON_BIN}" -c '
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
TORCH_INSTALL_DIR="${COMPONENT_ROOT}/core/multicore/shmem/lib/framework/torch"
rm -rf "${TORCH_INSTALL_DIR}"

# Build the SHMEM ops kernel library, then the HP SHMEM Runtime with its Torch bindings.
CCSRC_OPS_BUILD_DIR="${SHMEM_WORK_ROOT}/ccsrc-ops"
CURRENT_REASON_CODE="SHMEM_OPS_BUILD_FAILED"
cmake -S "${PROJECT_ROOT}/hyper_parallel/core/multicore/shmem/ccsrc/ops" \
    -B "${CCSRC_OPS_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OPS_INSTALL_DIR}"
cmake --build "${CCSRC_OPS_BUILD_DIR}" --parallel "${NATIVE_JOBS}"
cmake --install "${CCSRC_OPS_BUILD_DIR}"
SHMEM_OPS_LIB="${OPS_INSTALL_DIR}/lib/libhyper_parallel_shmem_ops.so"
if [[ ! -s "${SHMEM_OPS_LIB}" ]]; then
    fail "EXPECTED_ARTIFACT_MISSING" \
        "SHMEM ops kernel library was not installed: ${SHMEM_OPS_LIB}." 9
fi

CCSRC_TORCH_BUILD_DIR="${FRAMEWORK_WORK_ROOT}/ccsrc-torch-${TORCH_CACHE_KEY}"
CURRENT_REASON_CODE="CCSRC_TORCH_BINDING_BUILD_FAILED"
cmake -S "${PROJECT_ROOT}/hyper_parallel/core/multicore/shmem/ccsrc" \
    -B "${CCSRC_TORCH_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${TORCH_INSTALL_DIR}" \
    -DSHMEM_OPS_LIB="${SHMEM_OPS_LIB}" \
    -DPython3_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${CCSRC_TORCH_BUILD_DIR}" --parallel "${NATIVE_JOBS}"
cmake --install "${CCSRC_TORCH_BUILD_DIR}"
CCSRC_TORCH_MODULE="${TORCH_INSTALL_DIR}/hyper_parallel_shmem_torch.so"
if [[ ! -s "${CCSRC_TORCH_MODULE}" ]]; then
    fail "EXPECTED_ARTIFACT_MISSING" \
        "SHMEM Torch binding module was not installed: ${CCSRC_TORCH_MODULE}." 9
fi

CURRENT_REASON_CODE="SHMEM_ELF_VALIDATION_FAILED"
for private_library in \
    libhyper_parallel_shmem.so \
    libhyper_parallel_shmem_utils.so \
    aclshmem_bootstrap_uid.so \
    aclshmem_bootstrap_config_store.so; do
    private_library_path="${OPS_INSTALL_DIR}/lib/shmem/${private_library}"
    if [[ ! -s "${private_library_path}" ]]; then
        fail "EXPECTED_ARTIFACT_MISSING" "Private SHMEM library is missing: ${private_library_path}." 11
    fi
    if ! readelf -d "${private_library_path}" | grep -Eq "\(SONAME\).*\[${private_library}\]"; then
        fail "SHMEM_PRIVATE_SONAME_INVALID" \
            "Private SHMEM SONAME does not match ${private_library}: ${private_library_path}." 11
    fi
done

while IFS= read -r component_library; do
    dynamic_section=$(readelf -d "${component_library}")
    if grep -Eq '\(NEEDED\).*\[(libshmem|libshmem_utils|libshmem_bootstrap_[^]]*)\.so\]' \
        <<< "${dynamic_section}"; then
        fail "GENERIC_SHMEM_DT_NEEDED_FOUND" \
            "Component library depends on a generic SHMEM SONAME: ${component_library}." 11
    fi
    while IFS= read -r runpath_entry; do
        runpath_value=$(sed -n 's/.*\[\(.*\)\].*/\1/p' <<< "${runpath_entry}")
        IFS=':' read -r -a search_paths <<< "${runpath_value}"
        for search_path in "${search_paths[@]}"; do
            if [[ "${search_path}" == /* ]]; then
                fail "ABSOLUTE_RUNPATH_FOUND" \
                    "Component library contains an absolute RPATH/RUNPATH: ${component_library}: ${search_path}." 11
            fi
        done
    done < <(grep -E '(RPATH|RUNPATH)' <<< "${dynamic_section}" || true)
done < <(find "${OPS_INSTALL_DIR}" -type f -name '*.so' -print)

trap - ERR
"${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({
    "install_root": sys.argv[2], "toolchain_key": sys.argv[3]}))' \
    "${WORK_ROOT}/sdk.json" "${HP_SHMEM_INSTALL_ROOT}" "${HP_SHMEM_TOOLCHAIN_KEY}"
echo "INFO: SHMEM build completed"
echo "  framework: torch"
echo "  component: ${COMPONENT_ROOT}"
