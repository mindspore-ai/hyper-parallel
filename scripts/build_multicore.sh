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
PROJECT_ROOT=$(dirname "${SCRIPT_DIR}")
BUILD_DIR="${PROJECT_ROOT}/build"
NATIVE_ROOT="${BUILD_DIR}/native"
WORK_ROOT="${NATIVE_ROOT}/work/multicore"
COMPONENT_ROOT="${NATIVE_ROOT}/components/multicore/hyper_parallel"
OUTPUT_ROOT="${COMPONENT_ROOT}/core/multicore/lib"
PAYLOAD_COMPONENT_ROOT="${NATIVE_ROOT}/payload/hyper_parallel/core/multicore/lib"
PAYLOAD_STAGING_ROOT="${NATIVE_ROOT}/payload-staging/multicore.$$"
FRAMEWORK="all"
SOC_LIST="ascend910b,ascend910_93"
NATIVE_JOBS="$(nproc)"
CLEAN="off"
BUILD_MULTICORE_MINDSPORE=false
BUILD_MULTICORE_TORCH=false
CURRENT_REASON_CODE="MULTICORE_BUILD_FAILED"

function show_help() {
    cat <<EOF
Usage:
  bash scripts/build_multicore.sh [OPTIONS]

Options:
  --framework VALUE   Build framework adapters: mindspore, torch, or all. Default: all.
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

function validate_multicore_build_log() {
    local build_log=$1
    if grep -Eiq \
        '\[ERROR\]|ERROR REASON|CMake Error|fatal error:|:[0-9]+(:[0-9]+)?:[[:space:]]+error:|ninja: build stopped|(gmake|make)(\[[0-9]+\])?: \*\*\*' \
        "${build_log}"; then
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
            "Legacy or case-conflicting MegaMoe symbol found in ${library}." 12
    fi
    mapfile -t forbidden_files < <(find "${vendor_root}" -type f \
        \( -name '*.cpp' -o -name '*.h' -o -name '*.ini' -o -name '*.json' \
           -o -name '*.py' -o -name '*.txt' \) \
        -exec grep -El 'aclnnMegaMoe|aclnnHyperMegaMoE|HyperMegaMoE' {} +)
    if [[ ${#forbidden_files[@]} -ne 0 ]]; then
        fail "CANN_VENDOR_IDENTITY_CONFLICT" \
            "Legacy or case-conflicting MegaMoe identity found in ${forbidden_files[0]}." 12
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
            require_nonempty_artifact \
                "${vendor_root}/op_impl/ai_core/tbe/kernel/config/${soc}/${op_name}.json" \
                "${soc} ${op_name} kernel config"
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

case "${FRAMEWORK}" in
    all)
        BUILD_MULTICORE_MINDSPORE=true
        BUILD_MULTICORE_TORCH=true
        ;;
    mindspore)
        BUILD_MULTICORE_MINDSPORE=true
        ;;
    torch)
        BUILD_MULTICORE_TORCH=true
        ;;
esac

cd "${PROJECT_ROOT}"
if [[ "${CLEAN}" == "on" ]]; then
    rm -rf "${WORK_ROOT}" "${COMPONENT_ROOT}"
fi
rm -rf "${COMPONENT_ROOT}" "${PAYLOAD_COMPONENT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

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

required_tools=(awk asc_opc bash cmake find gcc g++ git grep ld ldd make nm python3 readelf readlink)
required_tools+=(sed sha256sum sort tee)
for required_tool in "${required_tools[@]}"; do
    if ! command -v "${required_tool}" >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" "Required multicore build tool not found on PATH: ${required_tool}." 4
    fi
done
PYTHON_CACHE_TAG=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
FRAMEWORK_WORK_ROOT="${WORK_ROOT}/framework/${PYTHON_CACHE_TAG}"
if ! python3 -c 'import setuptools, wheel' >/dev/null 2>&1; then
    fail "PYTHON_BUILD_DEPENDENCY_NOT_FOUND" \
        "The selected Python requires setuptools and wheel before compiling multicore." 5
fi
if [[ "${BUILD_MULTICORE_MINDSPORE}" == "true" ]]; then
    if ! command -v ninja >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" "Required MindSpore multicore build tool not found: ninja." 5
    fi
    if ! python3 -c 'import mindspore as ms; assert hasattr(ms.ops, "CustomOpBuilder")' >/dev/null 2>&1; then
        fail "MINDSPORE_CUSTOM_OP_BUILDER_NOT_FOUND" \
            "The selected Python must provide MindSpore CustomOpBuilder." 5
    fi
fi
if [[ "${BUILD_MULTICORE_TORCH}" == "true" ]]; then
    if ! python3 -c 'import torch, torch_npu' >/dev/null 2>&1; then
        fail "TORCH_BUILD_DEPENDENCY_NOT_FOUND" \
            "The selected Python must provide matching torch and torch_npu packages." 5
    fi
    if ! python3 -c 'import torch; raise SystemExit(0 if torch._C._GLIBCXX_USE_CXX11_ABI else 1)' \
        >/dev/null 2>&1; then
        fail "TORCH_CXX11_ABI_UNSUPPORTED" \
            "The selected Torch native build requires _GLIBCXX_USE_CXX11_ABI=1." 5
    fi
fi

source "${SCRIPT_DIR}/check_gcc_version.sh"
check_gcc_version || fail "UNSUPPORTED_GCC" "Host GCC is outside the supported build range." 6

OPS_NN_SOURCE_DIR="${NATIVE_ROOT}/deps/ops_nn/src"
OPS_TRANSFORMER_SOURCE_DIR="${NATIVE_ROOT}/deps/ops_transformer/src"
CANN_CMAKE_SOURCE_DIR="${NATIVE_ROOT}/deps/cann_cmake/src"
OPBASE_SOURCE_DIR="${NATIVE_ROOT}/deps/opbase/src"
OPS_TENSOR_SOURCE_DIR="${NATIVE_ROOT}/deps/ops_tensor/src"
ARCHIVE_DIR="${NATIVE_ROOT}/deps/multicore/archives"

CURRENT_REASON_CODE="MULTICORE_DEPENDENCY_PREPARATION_FAILED"
python3 -m scripts.native.prepare_dependencies \
    --dependency ops_nn \
    --dependency ops_transformer \
    --dependency cann_cmake \
    --dependency opbase \
    --dependency ops_tensor \
    --dependency multicore_archives

source "${SCRIPT_DIR}/native/shmem_sdk.sh"
CURRENT_REASON_CODE="SHMEM_SDK_BUILD_FAILED"
hp_prepare_shmem_sdk "${PROJECT_ROOT}" "${SOC_LIST}" "${NATIVE_JOBS}" "off"
SHMEM_INSTALL_ROOT="${HP_SHMEM_INSTALL_ROOT}"
if [[ ! -f "${SHMEM_INSTALL_ROOT}/shmem/include/shmem.h" \
      || ! -d "${SHMEM_INSTALL_ROOT}/shmem/src" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/libhyper_parallel_shmem.so" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/libhyper_parallel_shmem_utils.so" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/aclshmem_bootstrap_uid.so" \
      || ! -f "${SHMEM_INSTALL_ROOT}/shmem/lib/aclshmem_bootstrap_config_store.so" ]]; then
    fail "SHMEM_SDK_NOT_PREPARED" \
        "Full SHMEM SDK is missing at ${SHMEM_INSTALL_ROOT}." 9
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
                "arch=$(uname -m)"
            sha256sum "${CANN_VERSION_FILE}"
            sha256sum "${asc_opc_path}"
            sha256sum \
                scripts/build_multicore.sh \
                scripts/native/config/dependencies.lock.json \
                scripts/native/assemble_multicore_source.py \
                scripts/native/merge_multicore_vendors.py \
                scripts/native/shmem_sdk.sh
            while IFS= read -r -d '' source_file; do
                sha256sum "${source_file}"
            done < <(
                find hyper_parallel/core/multicore/ops scripts/native/cmake/shmem_wrapper \
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

    rm -rf "${ASSEMBLY_ROOT}"
    CURRENT_REASON_CODE="MULTICORE_SOURCE_ASSEMBLY_FAILED"
    python3 -m scripts.native.assemble_multicore_source \
        --ops-nn-source "${OPS_NN_SOURCE_DIR}" \
        --ops-transformer-source "${OPS_TRANSFORMER_SOURCE_DIR}" \
        --cann-cmake-source "${CANN_CMAKE_SOURCE_DIR}" \
        --opbase-source "${OPBASE_SOURCE_DIR}" \
        --ops-tensor-source "${OPS_TENSOR_SOURCE_DIR}" \
        --third-party-dir "${ARCHIVE_DIR}" \
        --work-dir "${ASSEMBLY_ROOT}"

    for op_name in hyper_mega_moe hyper_mega_moe_grad; do
        if [[ ! -d "${SOURCE_ROOT}/mega_moe/${op_name}/op_host/config/${cann_soc}" ]]; then
            fail "SOC_SOURCE_NOT_SUPPORTED" \
                "${op_name} has no ${cann_soc} config in the locked source; this SoC is not supported." 11
        fi
    done

    echo "INFO: building isolated ${cann_soc} vendor input from ${SOURCE_ROOT}"
    CURRENT_REASON_CODE="CANN_VENDOR_BUILD_FAILED"
    set +e
    (
        cd "${SOURCE_ROOT}"
        bash build.sh \
            --pkg \
            --soc="${cann_soc}" \
            --ops=hyper_mega_moe,hyper_mega_moe_grad \
            --vendor_name=hyper_parallel_multicore \
            --cann_3rd_lib_path="${SOURCE_ROOT}/third_party" \
            -j"${NATIVE_JOBS}"
    ) 2>&1 | tee "${KERNEL_LOG}"
    cann_build_status=${PIPESTATUS[0]}
    set -e
    if [[ ${cann_build_status} -ne 0 ]]; then
        fail "CANN_VENDOR_BUILD_FAILED" \
            "${cann_soc} CANN vendor build exited ${cann_build_status}; inspect ${KERNEL_LOG}." \
            "${cann_build_status}"
    fi

    mapfile -t vendor_candidates < <(
        find "${SOURCE_ROOT}/build_out/_CPack_Packages" -type d \
            -path "*/packages/vendors/hyper_parallel_multicore_nn" -print
    )
    if [[ ${#vendor_candidates[@]} -ne 1 ]]; then
        fail "UNIFIED_VENDOR_NOT_FOUND" \
            "Expected one ${cann_soc} hyper_parallel_multicore_nn package, found ${#vendor_candidates[@]}." 12
    fi
    CURRENT_REASON_CODE="CANN_VENDOR_VALIDATION_FAILED"
    validate_multicore_vendor "${vendor_candidates[0]}" "${cann_soc}" "${KERNEL_LOG}"
    rm -rf "${CACHE_ROOT}"
    mkdir -p "${CACHE_ROOT}"
    cp -a "${vendor_candidates[0]}" "${SNAPSHOT_VENDOR}"
    printf '%s\n' "${VENDOR_FINGERPRINT}" > "${FINGERPRINT_FILE}"
    printf '%s\n' "${COMMON_HOST_INPUT_IDENTITY}" > "${HOST_INPUT_IDENTITY_FILE}"
    PER_SOC_VENDOR_INPUTS+=("--input" "${cann_soc}=${SNAPSHOT_VENDOR}")
    PER_SOC_HOST_INPUT_IDENTITIES+=("--host-input-identity" "${cann_soc}=${COMMON_HOST_INPUT_IDENTITY}")
done

VENDOR_ROOT="${OUTPUT_ROOT}/vendors/hyper_parallel_multicore_nn"
mkdir -p "$(dirname "${VENDOR_ROOT}")"
CURRENT_REASON_CODE="CANN_VENDOR_MERGE_FAILED"
python3 -m scripts.native.merge_multicore_vendors \
    "${PER_SOC_VENDOR_INPUTS[@]}" \
    "${PER_SOC_HOST_INPUT_IDENTITIES[@]}" \
    --output "${VENDOR_ROOT}"

CURRENT_REASON_CODE="CANN_VENDOR_VALIDATION_FAILED"
validate_multicore_vendor "${VENDOR_ROOT}" "${CANN_SOC_LIST}"
cp "${PROJECT_ROOT}/hyper_parallel/core/multicore/set_env.bash" "${OUTPUT_ROOT}/set_env.bash"

export HP_MULTICORE_VENDOR_ROOT="${VENDOR_ROOT}"
export CANN_VENDOR_LIBDIR="${VENDOR_ROOT}/op_api/lib"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "${BUILD_MULTICORE_MINDSPORE}" == "true" ]]; then
    CURRENT_REASON_CODE="MINDSPORE_ADAPTER_BUILD_FAILED"
    MINDSPORE_SOURCE="${PROJECT_ROOT}/hyper_parallel/core/multicore/platform/mindspore"
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
    MINDSPORE_BUILD="${FRAMEWORK_WORK_ROOT}/mindspore-${MINDSPORE_CACHE_KEY}"
    MINDSPORE_OUTPUT="${OUTPUT_ROOT}/framework/mindspore"
    rm -rf "${MINDSPORE_BUILD}" "${MINDSPORE_OUTPUT}"
    cmake -S "${MINDSPORE_SOURCE}" \
        -B "${MINDSPORE_BUILD}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DHP_MULTICORE_VENDOR_ROOT="${VENDOR_ROOT}" \
        -DPython3_EXECUTABLE="$(command -v python3)"
    cmake --build "${MINDSPORE_BUILD}" --parallel "${NATIVE_JOBS}"
    if [[ ! -s "${MINDSPORE_BUILD}/lib/hyper_parallel_mega_moe_ms.so" ]]; then
        fail "EXPECTED_ARTIFACT_MISSING" \
            "MindSpore CustomOpBuilder returned without producing hyper_parallel_mega_moe_ms.so." 13
    fi
    mkdir -p "${MINDSPORE_OUTPUT}"
    cp -a "${MINDSPORE_BUILD}/lib/hyper_parallel_mega_moe_ms.so" "${MINDSPORE_OUTPUT}/"
    if [[ -d "${MINDSPORE_BUILD}/lib/hyper_parallel_mega_moe_ms_auto_generate" ]]; then
        cp -a "${MINDSPORE_BUILD}/lib/hyper_parallel_mega_moe_ms_auto_generate" "${MINDSPORE_OUTPUT}/"
    fi
fi

if [[ "${BUILD_MULTICORE_TORCH}" == "true" ]]; then
    CURRENT_REASON_CODE="TORCH_ADAPTER_BUILD_FAILED"
    TORCH_SOURCE="${PROJECT_ROOT}/hyper_parallel/core/multicore/platform/torch"
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
    TORCH_BUILD="${FRAMEWORK_WORK_ROOT}/torch-${TORCH_CACHE_KEY}"
    TORCH_OUTPUT="${OUTPUT_ROOT}/framework/torch"
    rm -rf "${TORCH_BUILD}" "${TORCH_OUTPUT}"
    mkdir -p "${TORCH_BUILD}/lib" "${TORCH_BUILD}/temp" "${TORCH_OUTPUT}"
    (
        cd "${TORCH_SOURCE}"
        python3 setup.py build_ext \
            --build-lib "${TORCH_BUILD}/lib" \
            --build-temp "${TORCH_BUILD}/temp"
    )
    mapfile -t torch_adapters < <(
        find "${TORCH_BUILD}/lib" -maxdepth 1 -type f \
            -name 'hyper_parallel_mega_moe_pta*.so' -print
    )
    if [[ ${#torch_adapters[@]} -ne 1 ]]; then
        fail "TORCH_ADAPTER_ARTIFACT_MISSING" \
            "Expected one PyTorch multicore adapter, found ${#torch_adapters[@]}." 13
    fi
    cp -a "${torch_adapters[0]}" "${TORCH_OUTPUT}/"
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
            libmindspore_*.so|libmindspore_*.so.*|libtorch*.so|libc10*.so)
                ;;
            *)
                fail "HOST_ELF_DEPENDENCY_NOT_FOUND" \
                    "Host adapter dependency is unavailable outside framework import: ${unresolved_library}." 19
                ;;
        esac
    done
done < <(find "${OUTPUT_ROOT}/framework" -type f -name '*.so' -print)

rm -rf "${PAYLOAD_STAGING_ROOT}"
mkdir -p "$(dirname "${PAYLOAD_STAGING_ROOT}")" "$(dirname "${PAYLOAD_COMPONENT_ROOT}")"
cp -a "${OUTPUT_ROOT}" "${PAYLOAD_STAGING_ROOT}"
mv "${PAYLOAD_STAGING_ROOT}" "${PAYLOAD_COMPONENT_ROOT}"

trap - ERR
echo "INFO: multicore build completed"
echo "  framework: ${FRAMEWORK}"
echo "  vendor: ${VENDOR_ROOT}"
echo "  framework payload: ${OUTPUT_ROOT}/framework"
echo "  component: ${COMPONENT_ROOT}"
echo "  PYTHONPATH payload: ${PAYLOAD_COMPONENT_ROOT}"
