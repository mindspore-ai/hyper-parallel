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
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${PROJECT_ROOT}/scripts/native/cann_version.sh"
HP_MINIMUM_CANN_VERSION="9.1.0"
BUILD_DIR="${PROJECT_ROOT}/build"
NATIVE_ROOT="${BUILD_DIR}/native"
WORK_ROOT="${NATIVE_ROOT}/work/multicore"
COMPONENT_ROOT="${NATIVE_ROOT}/components/multicore/hyper_parallel"
OUTPUT_ROOT="${COMPONENT_ROOT}/core/multicore/lib"
SOC_LIST="ascend910b,ascend910_93"
NATIVE_JOBS="$(nproc)"
CLEAN="off"
STAGE_ONLY="off"
CURRENT_REASON_CODE="MULTICORE_BUILD_FAILED"
PYTHON_BIN=python

function show_help() {
    cat <<EOF
Usage:
  bash hyper_parallel/core/multicore/build.sh [OPTIONS]

Options:
  --soc-list VALUE    Comma-separated CANN SoC IDs. Default: ascend910b,ascend910_93.
  --jobs VALUE        Parallel build jobs. Default: nproc.
  --clean             Remove multicore work and install outputs before building.
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
            "${key_name} cache identity must be one 16-character hexadecimal digest." 5
    fi
}

function report_unhandled_error() {
    local exit_code=$?
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${CURRENT_REASON_CODE}"
    echo "ERROR: multicore build failed unexpectedly with exit ${exit_code}." >&2
    exit "${exit_code}"
}
trap report_unhandled_error ERR

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
        --stage-only)
            STAGE_ONLY="on"
            shift
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

RAW_SOC_LIST="${SOC_LIST}"
IFS=',' read -r -a requested_socs <<< "${RAW_SOC_LIST}"
declare -a CANN_SOCS=()
declare -A seen_cann_socs=()
for requested_soc in "${requested_socs[@]}"; do
    requested_soc="${requested_soc//[[:space:]]/}"
    case "${requested_soc}" in
        ascend910b)
            cann_soc="ascend910b"
            ;;
        ascend910_93)
            cann_soc="ascend910_93"
            ;;
        ascend950)
            cann_soc="ascend950"
            ;;
        *)
            soc_error="Unsupported SoC '${requested_soc}' in --soc-list=${RAW_SOC_LIST}; "
            soc_error+="use ascend910b, ascend910_93, or ascend950."
            fail "UNSUPPORTED_SOC_SELECTION" "${soc_error}" 10
            ;;
    esac
    if [[ -z "${seen_cann_socs[${cann_soc}]:-}" ]]; then
        CANN_SOCS+=("${cann_soc}")
        seen_cann_socs["${cann_soc}"]=1
    fi
done
if [[ ${#CANN_SOCS[@]} -eq 0 ]]; then
    fail "UNSUPPORTED_SOC_SELECTION" "--soc-list must select at least one SoC." 10
fi
CANN_SOC_LIST=$(IFS=,; echo "${CANN_SOCS[*]}")

function validate_multicore_build_log() {
    local build_log=$1
    if grep -Eiv '^\[ERROR\] ATRACE\(.*get platform info failed, drvErr=4\.$' "${build_log}" | \
            grep -Eiq \
                '\[ERROR\]|ERROR REASON|CMake Error|fatal error:|:[0-9]+(:[0-9]+)?:[[:space:]]+error:|ninja: build stopped|(gmake|make)(\[[0-9]+\])?: \*\*\*'; then
        fail "CANN_VENDOR_BUILD_LOG_ERROR" \
            "Fatal compiler or CANN message found in ${build_log}." 12
    fi
}

function require_nonempty_artifact() {
    local artifact=$1
    local description=$2
    if [[ ! -s "${artifact}" ]]; then
        fail "CANN_VENDOR_ARTIFACT_MISSING" \
            "Missing or empty ${description}: ${artifact}." 12
    fi
}

function validate_multicore_vendor() {
    local vendor_root=$1
    local soc_list=$2
    local build_log=${3:-}
    local library
    local dynamic_section
    local ldd_output
    local op_name
    local required_symbol
    local soc
    local artifact
    local runpath_entry
    local runpath_value
    local search_path
    local symbol_table
    local -a forbidden_files=()
    local -a libraries=()
    local -a artifacts=()
    local -a search_paths=()
    local -a validation_socs=()
    local -a required_symbols=(
        aclnnHyperMegaMoe
        aclnnHyperMegaMoeGetWorkspaceSize
        aclnnHyperMegaMoeGrad
        aclnnHyperMegaMoeGradGetWorkspaceSize
    )

    if [[ -n "${build_log}" ]]; then
        validate_multicore_build_log "${build_log}"
    fi
    mapfile -t libraries < <(find "${vendor_root}" -type f -name 'libcust_opapi.so' -print)
    if [[ ${#libraries[@]} -ne 1 ]]; then
        fail "CANN_VENDOR_LIBRARY_INVALID" \
            "Expected one libcust_opapi.so under ${vendor_root}, found ${#libraries[@]}." 12
    fi
    library=${libraries[0]}
    symbol_table=$(nm -D --defined-only "${library}" | awk '{print $NF}')
    for required_symbol in "${required_symbols[@]}"; do
        if ! grep -Fxq "${required_symbol}" <<< "${symbol_table}"; then
            fail "CANN_VENDOR_SYMBOL_MISSING" \
                "Required symbol ${required_symbol} is missing from ${library}." 12
        fi
    done
    if grep -Eq 'aclnnMegaMoe|aclnnHyperMegaMoE|HyperMegaMoE' <<< "${symbol_table}"; then
        fail "CANN_VENDOR_SYMBOL_CONFLICT" \
            "Forbidden or case-conflicting MegaMoe symbol found in ${library}." 12
    fi
    mapfile -t forbidden_files < <(find "${vendor_root}" -type f \
        \( -name '*.cpp' -o -name '*.h' -o -name '*.ini' -o -name '*.json' \
           -o -name '*.py' -o -name '*.txt' \) \
        -exec grep -El 'aclnnMegaMoe|aclnnHyperMegaMoE|HyperMegaMoE' {} +)
    if [[ ${#forbidden_files[@]} -ne 0 ]]; then
        fail "CANN_VENDOR_IDENTITY_CONFLICT" \
            "Forbidden or case-conflicting MegaMoe identity found in ${forbidden_files[0]}." 12
    fi

    IFS=',' read -r -a validation_socs <<< "${soc_list}"
    for soc in "${validation_socs[@]}"; do
        for op_name in hyper_mega_moe hyper_mega_moe_grad; do
            mapfile -t artifacts < <(
                find "${vendor_root}/op_impl/ai_core/tbe/kernel/${soc}/${op_name}" \
                    -maxdepth 1 -type f -name '*.o' -print 2>/dev/null
            )
            if [[ ${#artifacts[@]} -eq 0 ]]; then
                fail "CANN_VENDOR_ARTIFACT_MISSING" \
                    "Missing ${soc} ${op_name} kernel object under ${vendor_root}." 12
            fi
            for artifact in "${artifacts[@]}"; do
                require_nonempty_artifact "${artifact}" "${soc} ${op_name} kernel object"
            done
            mapfile -t artifacts < <(
                find "${vendor_root}/op_impl/ai_core/tbe/kernel/${soc}/${op_name}" \
                    -maxdepth 1 -type f -name '*.json' -print 2>/dev/null
            )
            if [[ ${#artifacts[@]} -eq 0 ]]; then
                fail "CANN_VENDOR_ARTIFACT_MISSING" \
                    "Missing ${soc} ${op_name} kernel metadata under ${vendor_root}." 12
            fi
            for artifact in "${artifacts[@]}"; do
                require_nonempty_artifact "${artifact}" "${soc} ${op_name} kernel metadata"
            done
        done
        require_nonempty_artifact \
            "${vendor_root}/op_impl/ai_core/tbe/config/${soc}/aic-${soc}-ops-info.json" \
            "${soc} CANN op info config"
        require_nonempty_artifact \
            "${vendor_root}/op_impl/ai_core/tbe/kernel/config/${soc}/binary_info_config.json" \
            "${soc} binary index"
        for op_name in hyper_mega_moe hyper_mega_moe_grad; do
            require_nonempty_artifact \
                "${vendor_root}/op_impl/ai_core/tbe/kernel/config/${soc}/${op_name}.json" \
                "${soc} ${op_name} binary config"
        done
    done

    dynamic_section=$(readelf -d "${library}")
    if ! grep -Eq '\(SONAME\).*\[libcust_opapi\.so\]' <<< "${dynamic_section}"; then
        fail "CANN_VENDOR_SONAME_INVALID" \
            "Expected libcust_opapi.so SONAME in ${library}." 12
    fi
    while IFS= read -r runpath_entry; do
        runpath_value=$(sed -n 's/.*\[\(.*\)\].*/\1/p' <<< "${runpath_entry}")
        IFS=':' read -r -a search_paths <<< "${runpath_value}"
        for search_path in "${search_paths[@]}"; do
            if [[ "${search_path}" == /* ]]; then
                fail "CANN_VENDOR_ABSOLUTE_RUNPATH_FOUND" \
                    "Build-machine absolute RPATH/RUNPATH found in ${library}: ${search_path}." 12
            fi
        done
    done < <(grep -E '(RPATH|RUNPATH)' <<< "${dynamic_section}" || true)
    ldd_output=$(ldd -r "${library}" 2>&1)
    if grep -Eq 'not found|undefined symbol:' <<< "${ldd_output}"; then
        fail "CANN_VENDOR_RUNTIME_LINK_FAILED" \
            "Unresolved runtime dependency found for ${library}." 12
    fi
    echo "INFO: validated multicore vendor ${vendor_root} for ${soc_list}"
}

cd "${PROJECT_ROOT}"
if [[ "${CLEAN}" == "on" ]]; then
    rm -rf "${WORK_ROOT}" "${COMPONENT_ROOT}"
fi
rm -rf "${COMPONENT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
SHMEM_ARGS=(--soc-list "${SOC_LIST}" --jobs "${NATIVE_JOBS}")
if [[ "${CLEAN}" == "on" ]]; then
    SHMEM_ARGS+=(--clean)
fi
bash "${SCRIPT_DIR}/shmem/build.sh" "${SHMEM_ARGS[@]}"

if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
    fail "CANN_ENV_NOT_CONFIGURED" \
        "ASCEND_HOME_PATH must identify the selected CANN installation; source its set_env.sh first." 3
fi
CANN_VERSION_FILE="${ASCEND_HOME_PATH}/opp/version.info"
CANN_VERSION=$(hp_read_cann_version "${ASCEND_HOME_PATH}")
if ! hp_cann_version_at_least "${CANN_VERSION}" "${HP_MINIMUM_CANN_VERSION}"; then
    fail "UNSUPPORTED_CANN_VERSION" \
        "CANN >= ${HP_MINIMUM_CANN_VERSION} is required, found " \
        "'${CANN_VERSION:-unknown}' under ${ASCEND_HOME_PATH}." 3
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "PYTHON_BUILD_DEPENDENCY_NOT_FOUND" \
        "The selected Python executable is unavailable: ${PYTHON_BIN}." 4
fi
PYTHON_BIN=$(command -v "${PYTHON_BIN}")
required_tools=(awk asc_opc bash cmake find gcc g++ git grep ld ldd make nm readelf readlink)
required_tools+=(sed sha256sum sort tee)
for required_tool in "${required_tools[@]}"; do
    if ! command -v "${required_tool}" >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" "Required multicore build tool not found on PATH: ${required_tool}." 4
    fi
done
PYTHON_CACHE_TAG=$("${PYTHON_BIN}" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
FRAMEWORK_WORK_ROOT="${WORK_ROOT}/framework/${PYTHON_CACHE_TAG}"
    if ! TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON_BIN}" -c '
from importlib.util import find_spec
import torch
raise SystemExit(0 if find_spec("torch_npu") is not None else 1)
' \
        >/dev/null 2>&1; then
        fail "TORCH_BUILD_DEPENDENCY_NOT_FOUND" \
            "The selected Python must provide matching torch and torch_npu packages." 5
    fi

source "${PROJECT_ROOT}/scripts/check_gcc_version.sh"
check_gcc_version || fail "UNSUPPORTED_GCC" "Host GCC is outside the supported build range." 6

OPS_NN_SOURCE_DIR="${NATIVE_ROOT}/deps/ops_nn/src"
OPS_TRANSFORMER_SOURCE_DIR="${NATIVE_ROOT}/deps/ops_transformer/src"
MULTICORE_VENDOR_CMAKE="${PROJECT_ROOT}/hyper_parallel/core/multicore/cmake/vendor"

CURRENT_REASON_CODE="MULTICORE_DEPENDENCY_PREPARATION_FAILED"
"${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_parallel/core/multicore/_build/prepare_dependencies.py" \
    --dependency ops_nn \
    --dependency ops_transformer

CURRENT_REASON_CODE="SHMEM_SDK_HANDOFF_FAILED"
SHMEM_INSTALL_ROOT=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["install_root"])' \
    "${WORK_ROOT}/shmem/sdk.json")
HP_SHMEM_TOOLCHAIN_KEY=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["toolchain_key"])' \
    "${WORK_ROOT}/shmem/sdk.json")
if [[ ! -f "${SHMEM_INSTALL_ROOT}/shmem/include/shmem.h" \
      || ! -d "${SHMEM_INSTALL_ROOT}/shmem/src" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/libhyper_parallel_shmem.so" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/libhyper_parallel_shmem_utils.so" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/aclshmem_bootstrap_uid.so" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/aclshmem_bootstrap_config_store.so" ]]; then
    fail "SHMEM_SDK_NOT_PREPARED" \
        "Full SHMEM SDK is missing at ${SHMEM_INSTALL_ROOT}." 9
fi

if ! MULTICORE_LOCK_DIGEST=$("${PYTHON_BIN}" -c '
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    lock = json.load(source)["components"]["multicore"]
encoded = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
' "${PROJECT_ROOT}/hyper_parallel/core/multicore/_build/dependencies.lock.json"); then
    fail "MULTICORE_LOCK_INVALID" "Cannot read the locked multicore dependency contract." 9
fi
if [[ ! "${MULTICORE_LOCK_DIGEST}" =~ ^[0-9a-f]{64}$ ]]; then
    fail "MULTICORE_LOCK_INVALID" "Cannot read the locked multicore dependency contract." 9
fi

mkdir -p "${WORK_ROOT}/logs"

export ASCEND_SHMEM_HOME_PATH="${SHMEM_INSTALL_ROOT}"
export SHMEM_HOME_PATH="${SHMEM_INSTALL_ROOT}"
export PIP_NO_BUILD_ISOLATION=false
export PIP_NO_INDEX=1

echo "INFO: building unified multicore CANN vendor"
echo "  CANN: ${ASCEND_HOME_PATH}"
echo "  SOCs: ${CANN_SOC_LIST}"
echo "  SHMEM SDK: ${SHMEM_INSTALL_ROOT}"

function calculate_vendor_fingerprint() {
    local cann_soc=$1
    local cann_root
    local bisheng_path
    local bisheng_version
    local asc_opc_path
    local gcc_path
    local gxx_path
    local ld_path
    local cmake_path
    local make_path
    cann_root=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    bisheng_path=$(readlink -f "$(command -v bisheng)")
    bisheng_version=$("${bisheng_path}" --version 2>&1 | sed -n '1p')
    asc_opc_path=$(readlink -f "$(command -v asc_opc)")
    gcc_path=$(readlink -f "$(command -v gcc)")
    gxx_path=$(readlink -f "$(command -v g++)")
    ld_path=$(readlink -f "$(command -v ld)")
    cmake_path=$(readlink -f "$(command -v cmake)")
    make_path=$(readlink -f "$(command -v make)")
    (
        cd "${PROJECT_ROOT}"
        {
            printf '%s\n' \
                "schema=1" \
                "soc=${cann_soc}" \
                "cann=${CANN_VERSION}" \
                "cann_root=${cann_root}" \
                "bisheng=${bisheng_path}:${bisheng_version}" \
                "asc_opc=${asc_opc_path}" \
                "gcc=${gcc_path}:$(gcc -dumpmachine):$(gcc -dumpfullversion -dumpversion)" \
                "gxx=${gxx_path}:$(g++ -dumpmachine):$(g++ -dumpfullversion -dumpversion)" \
                "ld=${ld_path}:$(ld --version | sed -n '1p')" \
                "cmake=${cmake_path}:$(cmake --version | sed -n '1p')" \
                "make=${make_path}:$(make --version | sed -n '1p')" \
                "env_CC=${CC-}" \
                "env_CXX=${CXX-}" \
                "env_CFLAGS=${CFLAGS-}" \
                "env_CXXFLAGS=${CXXFLAGS-}" \
                "env_CPPFLAGS=${CPPFLAGS-}" \
                "env_LDFLAGS=${LDFLAGS-}" \
                "env_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE-}" \
                "multicore_lock=${MULTICORE_LOCK_DIGEST}" \
                "shmem_toolchain=${HP_SHMEM_TOOLCHAIN_KEY}" \
                "arch=$(uname -m)"
            sha256sum "${CANN_VERSION_FILE}"
            sha256sum "${asc_opc_path}"
            sha256sum \
                hyper_parallel/core/multicore/build.sh \
                hyper_parallel/core/multicore/_build/assemble_multicore_source.py \
                hyper_parallel/core/multicore/_build/merge_multicore_vendors.py \
                hyper_parallel/core/multicore/shmem/_build/shmem_sdk.sh
            while IFS= read -r -d '' source_file; do
                sha256sum "${source_file}"
            done < <(
                find hyper_parallel/core/multicore/ops hyper_parallel/core/multicore/cmake/vendor \
                    hyper_parallel/core/multicore/shmem/cmake/shmem_wrapper \
                    -type f -print0 | sort -z
            )
        } | sha256sum | awk '{print $1}'
    )
}

COMMON_HOST_INPUT_IDENTITY=$(calculate_vendor_fingerprint "common-host-input")
declare -a PER_SOC_VENDOR_INPUTS=()
declare -a PER_SOC_HOST_INPUT_IDENTITIES=()
for cann_soc in "${CANN_SOCS[@]}"; do
    ASSEMBLY_ROOT="${WORK_ROOT}/source-assembly/${cann_soc}"
    SOURCE_ROOT="${ASSEMBLY_ROOT}/source"
    VENDOR_BUILD_ROOT="${WORK_ROOT}/vendor-build/${cann_soc}"
    VENDOR_STAGE_ROOT="${VENDOR_BUILD_ROOT}/stage"
    VENDOR_CANDIDATE="${VENDOR_STAGE_ROOT}/packages/vendors/hyper_parallel_multicore_nn"
    KERNEL_LOG="${WORK_ROOT}/logs/cann-vendor-build-${cann_soc}.log"
    CACHE_ROOT="${WORK_ROOT}/vendor-cache/${cann_soc}"
    SNAPSHOT_VENDOR="${CACHE_ROOT}/hyper_parallel_multicore_nn"
    FINGERPRINT_FILE="${CACHE_ROOT}/fingerprint.sha256"
    HOST_INPUT_IDENTITY_FILE="${CACHE_ROOT}/common-host-input-identity.sha256"
    VENDOR_FINGERPRINT=$(calculate_vendor_fingerprint "${cann_soc}")
    if [[ -f "${FINGERPRINT_FILE}" \
          && -f "${HOST_INPUT_IDENTITY_FILE}" \
          && "$(<"${FINGERPRINT_FILE}")" == "${VENDOR_FINGERPRINT}" \
          && "$(<"${HOST_INPUT_IDENTITY_FILE}")" == "${COMMON_HOST_INPUT_IDENTITY}" \
          && -d "${SNAPSHOT_VENDOR}" ]]; then
        CURRENT_REASON_CODE="CANN_VENDOR_CACHE_VALIDATION_FAILED"
        validate_multicore_vendor "${SNAPSHOT_VENDOR}" "${cann_soc}"
        echo "INFO: reusing cached ${cann_soc} multicore vendor"
        PER_SOC_VENDOR_INPUTS+=("--input" "${cann_soc}=${SNAPSHOT_VENDOR}")
        PER_SOC_HOST_INPUT_IDENTITIES+=("--host-input-identity" "${cann_soc}=${COMMON_HOST_INPUT_IDENTITY}")
        continue
    fi

    rm -rf "${ASSEMBLY_ROOT}" "${VENDOR_BUILD_ROOT}"
    CURRENT_REASON_CODE="MULTICORE_SOURCE_ASSEMBLY_FAILED"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_parallel/core/multicore/_build/assemble_multicore_source.py" \
        --ops-nn-source "${OPS_NN_SOURCE_DIR}" \
        --ops-transformer-source "${OPS_TRANSFORMER_SOURCE_DIR}" \
        --work-dir "${ASSEMBLY_ROOT}"

    echo "INFO: building isolated ${cann_soc} vendor input from ${SOURCE_ROOT}"
    CURRENT_REASON_CODE="CANN_VENDOR_BUILD_FAILED"
    set +e
    (
        set -e
        cmake -S "${MULTICORE_VENDOR_CMAKE}" \
            -B "${VENDOR_BUILD_ROOT}" \
            -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
            -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
            -DASCEND_COMPUTE_UNIT="${cann_soc}" \
            -DASCEND_PYTHON_EXECUTABLE="${PYTHON_BIN}" \
            -DHP_MULTICORE_SOURCE_ROOT="${SOURCE_ROOT}" \
            -DHP_SHMEM_SDK_ROOT="${SHMEM_INSTALL_ROOT}/shmem" \
            -DHP_VENDOR_STAGE_ROOT="${VENDOR_STAGE_ROOT}"
        cmake --build "${VENDOR_BUILD_ROOT}" --parallel "${NATIVE_JOBS}"
        cmake --build "${VENDOR_BUILD_ROOT}" --parallel "${NATIVE_JOBS}" \
            --target binary
        cmake --install "${VENDOR_BUILD_ROOT}"
    ) 2>&1 | tee "${KERNEL_LOG}"
    cann_build_status=${PIPESTATUS[0]}
    set -e
    if [[ ${cann_build_status} -ne 0 ]]; then
        fail "CANN_VENDOR_BUILD_FAILED" \
            "${cann_soc} CANN vendor build exited ${cann_build_status}; inspect ${KERNEL_LOG}." \
            "${cann_build_status}"
    fi

    if [[ ! -d "${VENDOR_CANDIDATE}" ]]; then
        fail "UNIFIED_VENDOR_NOT_FOUND" \
            "Expected ${cann_soc} hyper_parallel_multicore_nn package at ${VENDOR_CANDIDATE}." 12
    fi
    CURRENT_REASON_CODE="CANN_VENDOR_VALIDATION_FAILED"
    validate_multicore_vendor "${VENDOR_CANDIDATE}" "${cann_soc}" "${KERNEL_LOG}"
    rm -rf "${CACHE_ROOT}"
    mkdir -p "${CACHE_ROOT}"
    cp -a "${VENDOR_CANDIDATE}" "${SNAPSHOT_VENDOR}"
    printf '%s\n' "${VENDOR_FINGERPRINT}" > "${FINGERPRINT_FILE}"
    printf '%s\n' "${COMMON_HOST_INPUT_IDENTITY}" > "${HOST_INPUT_IDENTITY_FILE}"
    PER_SOC_VENDOR_INPUTS+=("--input" "${cann_soc}=${SNAPSHOT_VENDOR}")
    PER_SOC_HOST_INPUT_IDENTITIES+=("--host-input-identity" "${cann_soc}=${COMMON_HOST_INPUT_IDENTITY}")
done

VENDOR_ROOT="${OUTPUT_ROOT}/vendors/hyper_parallel_multicore_nn"
mkdir -p "$(dirname "${VENDOR_ROOT}")"
CURRENT_REASON_CODE="CANN_VENDOR_MERGE_FAILED"
"${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_parallel/core/multicore/_build/merge_multicore_vendors.py" \
    "${PER_SOC_VENDOR_INPUTS[@]}" \
    "${PER_SOC_HOST_INPUT_IDENTITIES[@]}" \
    --output "${VENDOR_ROOT}"

CURRENT_REASON_CODE="CANN_VENDOR_VALIDATION_FAILED"
validate_multicore_vendor "${VENDOR_ROOT}" "${CANN_SOC_LIST}"
cp "${PROJECT_ROOT}/hyper_parallel/core/multicore/set_env.bash" "${OUTPUT_ROOT}/set_env.bash"

export HP_MULTICORE_VENDOR_ROOT="${VENDOR_ROOT}"
export CANN_VENDOR_LIBDIR="${VENDOR_ROOT}/op_api/lib"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CURRENT_REASON_CODE="TORCH_ADAPTER_BUILD_FAILED"
TORCH_SOURCE="${PROJECT_ROOT}/hyper_parallel/core/multicore/torch"
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
TORCH_BUILD="${FRAMEWORK_WORK_ROOT}/torch-${TORCH_CACHE_KEY}"
TORCH_OUTPUT="${OUTPUT_ROOT}/framework/torch"
rm -rf "${TORCH_BUILD}" "${TORCH_OUTPUT}"
mkdir -p "${TORCH_BUILD}" "${TORCH_OUTPUT}"
cmake -S "${TORCH_SOURCE}" \
    -B "${TORCH_BUILD}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${TORCH_OUTPUT}" \
    -DHP_MULTICORE_VENDOR_ROOT="${VENDOR_ROOT}" \
    -DPython3_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${TORCH_BUILD}" --parallel "${NATIVE_JOBS}"
cmake --install "${TORCH_BUILD}"
if [[ ! -s "${TORCH_OUTPUT}/libhyper_parallel_mega_moe_torch.so" ]]; then
    fail "TORCH_ADAPTER_ARTIFACT_MISSING" \
        "PyTorch multicore adapter was not installed under ${TORCH_OUTPUT}." 13
fi

CURRENT_REASON_CODE="HOST_ELF_VALIDATION_FAILED"
case "$(uname -m)" in
    aarch64|arm64)
        EXPECTED_ELF_MACHINE="AArch64"
        ;;
    x86_64|amd64)
        EXPECTED_ELF_MACHINE="Advanced Micro Devices X86-64"
        ;;
    *)
        fail "UNSUPPORTED_HOST_ARCHITECTURE" \
            "Unsupported multicore host architecture: $(uname -m)." 14
        ;;
esac
while IFS= read -r adapter_library; do
    dynamic_section=$(readelf -d "${adapter_library}")
    elf_machine=$(readelf -h "${adapter_library}" | awk -F: '/Machine:/{sub(/^[[:space:]]+/, "", $2); print $2}')
    if [[ "${elf_machine}" != "${EXPECTED_ELF_MACHINE}" ]]; then
        fail "HOST_ELF_ARCHITECTURE_MISMATCH" \
            "Host adapter machine '${elf_machine}' does not match '${EXPECTED_ELF_MACHINE}': ${adapter_library}." 15
    fi
    if ! grep -E '(RPATH|RUNPATH).*\$ORIGIN' <<< "${dynamic_section}" >/dev/null; then
        fail "RELATIVE_RUNPATH_MISSING" \
            "Host adapter has no component-relative \$ORIGIN RUNPATH: ${adapter_library}." 17
    fi
    while IFS= read -r runpath_entry; do
        runpath_value=$(sed -n 's/.*\[\(.*\)\].*/\1/p' <<< "${runpath_entry}")
        IFS=':' read -r -a search_paths <<< "${runpath_value}"
        for search_path in "${search_paths[@]}"; do
            if [[ "${search_path}" == /* ]]; then
                fail "ABSOLUTE_RUNPATH_FOUND" \
                    "Host adapter contains a build-machine RPATH/RUNPATH: ${adapter_library}: ${search_path}." 16
            fi
        done
    done < <(grep -E '(RPATH|RUNPATH)' <<< "${dynamic_section}" || true)
    if grep -E '\(NEEDED\).*libcust_opapi\.so' <<< "${dynamic_section}" >/dev/null; then
        fail "GENERIC_VENDOR_DT_NEEDED_FOUND" \
            "Host adapter directly depends on generic libcust_opapi.so: ${adapter_library}." 18
    fi
    mapfile -t unresolved_libraries < <(ldd "${adapter_library}" | awk '/not found/{print $1}')
    for unresolved_library in "${unresolved_libraries[@]}"; do
        case "${unresolved_library}" in
            libtorch*.so|libc10*.so)
                ;;
            *)
                fail "HOST_ELF_DEPENDENCY_NOT_FOUND" \
                    "Host adapter dependency is unavailable outside framework import: ${unresolved_library}." 19
                ;;
        esac
    done
done < <(find "${OUTPUT_ROOT}/framework" -type f -name '*.so' -print)

trap - ERR
echo "INFO: multicore build completed"
echo "  framework: torch"
echo "  vendor: ${VENDOR_ROOT}"
echo "  framework payload: ${OUTPUT_ROOT}/framework"
echo "  component: ${COMPONENT_ROOT}"

if [[ "${STAGE_ONLY}" == "off" ]]; then
    PAYLOAD_COMPONENT="${NATIVE_ROOT}/payload/hyper_parallel/core/multicore"
    rm -rf "${PAYLOAD_COMPONENT}"
    mkdir -p "${PAYLOAD_COMPONENT}"
    cp -a "${COMPONENT_ROOT}/core/multicore/." "${PAYLOAD_COMPONENT}/"
fi
