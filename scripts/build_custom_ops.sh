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
# Build the MindSpore custom-op adapter.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "${SCRIPT_DIR}")
CUSTOM_OPS_SRC="${PROJECT_ROOT}/hyper_parallel/platform/mindspore/custom_ops"
NATIVE_ROOT="${PROJECT_ROOT}/build/native"
WORK_ROOT="${NATIVE_ROOT}/work/custom_ops"
COMPONENT_ROOT="${NATIVE_ROOT}/components/custom_ops/hyper_parallel"
OUTPUT_DIR="${COMPONENT_ROOT}/platform/mindspore/custom_ops/lib"
PAYLOAD_OUTPUT_DIR="${NATIVE_ROOT}/payload/hyper_parallel/platform/mindspore/custom_ops/lib"
PAYLOAD_STAGING_ROOT="${NATIVE_ROOT}/payload-staging/custom_ops.$$"
FRAMEWORK="mindspore"
NATIVE_JOBS="$(nproc)"
CLEAN="off"
CURRENT_REASON_CODE="CUSTOM_OPS_BUILD_FAILED"

function show_help() {
    cat <<EOF
Usage:
  bash scripts/build_custom_ops.sh [OPTIONS]

Options:
  --framework VALUE   Build target: mindspore, torch, or all. Default: mindspore.
                      Only MindSpore produces a native adapter; torch reports a warning.
  --jobs VALUE        Parallel build jobs. Default: nproc.
  --clean             Remove custom-ops work and install outputs before building.
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
            "${key_name} cache identity must be one 16-character hexadecimal digest." 6
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

function report_unhandled_error() {
    local exit_code=$?
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${CURRENT_REASON_CODE}"
    echo "ERROR: custom_ops build failed unexpectedly with exit ${exit_code}." >&2
    exit "${exit_code}"
}
trap report_unhandled_error ERR

cd "${PROJECT_ROOT}"
if [[ "${CLEAN}" == "on" ]]; then
    rm -rf "${WORK_ROOT}" "${COMPONENT_ROOT}"
fi
rm -rf "${COMPONENT_ROOT}" "${PAYLOAD_OUTPUT_DIR}"

if [[ "${FRAMEWORK}" == "torch" || "${FRAMEWORK}" == "all" ]]; then
    echo "WARNING: [HP-NATIVE-FRAMEWORK-NOT-SUPPORTED] component=custom_ops framework=torch; " \
         "no Torch native custom-op adapter is produced."
fi
if [[ "${FRAMEWORK}" == "torch" ]]; then
    echo "INFO: custom ops build completed without a native payload for framework=torch."
    exit 0
fi

if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
    fail "CANN_ENV_NOT_CONFIGURED" \
        "ASCEND_HOME_PATH must identify the selected CANN installation; source its set_env.sh first." 3
fi
CANN_VERSION_FILE="${ASCEND_HOME_PATH}/opp/version.info"
CANN_VERSION=$(awk -F= '$1 == "Version" {print $2}' "${CANN_VERSION_FILE}" 2>/dev/null || true)
if [[ "${CANN_VERSION}" != "9.1.0" ]]; then
    fail "UNSUPPORTED_CANN_VERSION" \
        "CANN 9.1.0 is required, found '${CANN_VERSION:-unknown}' under ${ASCEND_HOME_PATH}." 3
fi
for cann_library in libascendcl.so libopapi.so; do
    if [[ ! -f "${ASCEND_HOME_PATH}/lib64/${cann_library}" ]]; then
        fail "CANN_LIBRARY_NOT_FOUND" \
            "Required CANN library is missing: ${ASCEND_HOME_PATH}/lib64/${cann_library}." 4
    fi
done
for required_tool in cmake gcc g++ grep make ninja python3 readelf sed; do
    if ! command -v "${required_tool}" >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" "Required custom_ops build tool not found on PATH: ${required_tool}." 5
    fi
done
PYTHON_CACHE_TAG=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
if ! python3 -c 'import mindspore as ms; assert hasattr(ms.ops, "CustomOpBuilder")' >/dev/null 2>&1; then
    fail "MINDSPORE_CUSTOM_OP_BUILDER_NOT_FOUND" \
        "The active Python environment must provide MindSpore CustomOpBuilder." 6
fi
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
PYTHON_WORK_ROOT="${WORK_ROOT}/${PYTHON_CACHE_TAG}/mindspore-${MINDSPORE_CACHE_KEY}"

source "${SCRIPT_DIR}/check_gcc_version.sh"
check_gcc_version || fail "UNSUPPORTED_GCC" "Host GCC is outside the supported build range." 7

if [[ ! -d "${CUSTOM_OPS_SRC}" ]]; then
    fail "CUSTOM_OPS_SOURCE_NOT_FOUND" "Source directory not found: ${CUSTOM_OPS_SRC}." 8
fi

rm -rf "${PYTHON_WORK_ROOT}"
mkdir -p "${PYTHON_WORK_ROOT}"
CURRENT_REASON_CODE="MINDSPORE_CUSTOM_OPS_BUILD_FAILED"
cmake -S "${CUSTOM_OPS_SRC}" \
    --no-warn-unused-cli \
    -B "${PYTHON_WORK_ROOT}" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "${PYTHON_WORK_ROOT}" --parallel "${NATIVE_JOBS}"

CUSTOM_OPS_LIBRARY="${PYTHON_WORK_ROOT}/lib/hyper_parallel_custom_ops_ms.so"
if [[ ! -s "${CUSTOM_OPS_LIBRARY}" ]]; then
    fail "EXPECTED_ARTIFACT_MISSING" \
        "MindSpore CustomOpBuilder returned without producing ${CUSTOM_OPS_LIBRARY}." 9
fi
dynamic_section=$(readelf -d "${CUSTOM_OPS_LIBRARY}")
while IFS= read -r runpath_entry; do
    runpath_value=$(sed -n 's/.*\[\(.*\)\].*/\1/p' <<< "${runpath_entry}")
    IFS=':' read -r -a search_paths <<< "${runpath_value}"
    for search_path in "${search_paths[@]}"; do
        if [[ "${search_path}" == /* ]]; then
            fail "ABSOLUTE_RUNPATH_FOUND" \
                "custom_ops adapter contains a build-machine RPATH/RUNPATH: ${CUSTOM_OPS_LIBRARY}: ${search_path}." 10
        fi
    done
done < <(grep -E '(RPATH|RUNPATH)' <<< "${dynamic_section}" || true)

mkdir -p "${OUTPUT_DIR}"
cp -a "${CUSTOM_OPS_LIBRARY}" "${OUTPUT_DIR}/"
if [[ -d "${PYTHON_WORK_ROOT}/lib/hyper_parallel_custom_ops_ms_auto_generate" ]]; then
    cp -a "${PYTHON_WORK_ROOT}/lib/hyper_parallel_custom_ops_ms_auto_generate" "${OUTPUT_DIR}/"
fi

rm -rf "${PAYLOAD_STAGING_ROOT}"
mkdir -p "$(dirname "${PAYLOAD_STAGING_ROOT}")" "$(dirname "${PAYLOAD_OUTPUT_DIR}")"
cp -a "${OUTPUT_DIR}" "${PAYLOAD_STAGING_ROOT}"
mv "${PAYLOAD_STAGING_ROOT}" "${PAYLOAD_OUTPUT_DIR}"

trap - ERR
echo "INFO: custom ops build completed"
echo "  framework: ${FRAMEWORK}"
echo "  component: ${COMPONENT_ROOT}"
echo "  PYTHONPATH payload: ${PAYLOAD_OUTPUT_DIR}"
