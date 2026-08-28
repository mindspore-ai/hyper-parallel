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
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT="${SCRIPT_DIR}"
NATIVE_ROOT="${PROJECT_ROOT}/build/native"
PAYLOAD_ROOT="${NATIVE_ROOT}/payload/hyper_parallel"
COMPONENT_ROOT="${NATIVE_ROOT}/components"
LOG_ROOT="${NATIVE_ROOT}/logs"
WHEEL_OUTPUT_ROOT="${NATIVE_ROOT}/wheel-output"
DEFAULT_CANN_SET_ENV="/usr/local/Ascend/cann/set_env.sh"

MULTICORE_VALUE="all"
SHMEM_VALUE="all"
CUSTOM_OPS_VALUE="on"
STRICT_VALUE="off"
SOC_LIST_VALUE="ascend910b,ascend910_93"
NATIVE_JOBS="$(nproc)"
CLEAN="off"

function show_help() {
    cat <<EOF
Usage:
  ./build.sh [OPTIONS]

The indexed Dataset C++ helpers are built on every invocation.

Options:
  --multicore VALUE    Build multicore: off, mindspore/ms, torch/pytorch, or all/both. Default: all.
  --shmem VALUE        Build symmetric memory: off, mindspore/ms, torch/pytorch, or all/both. Default: all.
                       Multicore automatically enables the matching symmetric-memory framework target.
  --custom-ops VALUE   Build custom ops: on or off. Default: on.
  --soc-list VALUE     Comma-separated CANN SoC IDs. Default: ascend910b,ascend910_93.
                       Supported: ascend910b and ascend910_93. ascend950 reports an optional native failure.
  --strict VALUE       Fail when an optional native component fails: on or off. Default: off.
  --jobs VALUE         Parallel native build jobs. Default: nproc.
  --clean              Remove selected component work/install caches before building.
  -h, --help           Show this help message.

The command always assembles the PYTHONPATH payload and creates a wheel. Native dependencies are checked and
downloaded automatically into build/native/deps when their locked versions are absent or inconsistent.
When ASCEND_HOME_PATH is unset, the default /usr/local/Ascend/cann/set_env.sh is sourced if it exists.

Examples:
  ./build.sh
  ./build.sh --multicore mindspore --shmem mindspore --custom-ops on
  ./build.sh --multicore torch --shmem torch --custom-ops off --strict off
  ./build.sh --multicore off --shmem off --custom-ops off
  ./build.sh --clean --jobs 32
EOF
}

function die() {
    echo "ERROR: $*" >&2
    echo "Run './build.sh --help' for usage." >&2
    exit 1
}

function require_value() {
    local option_name=$1
    local option_value=${2:-}
    if [[ -z "${option_value}" || "${option_value}" == --* ]]; then
        die "${option_name} requires a value."
    fi
}

function normalize_framework() {
    local value
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "${value}" in
        off)
            echo "off"
            ;;
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
            die "Unsupported framework selection '${1}'; use off, mindspore/ms, torch/pytorch, or all/both."
            ;;
    esac
}

function normalize_on_off() {
    local value
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "${value}" in
        on|off)
            echo "${value}"
            ;;
        *)
            die "Unsupported value '${1}'; use on or off."
            ;;
    esac
}

function validate_soc_list() {
    local soc_list=$1
    local requested_soc
    local -a requested_socs=()
    IFS=',' read -r -a requested_socs <<< "${soc_list}"
    if [[ ${#requested_socs[@]} -eq 0 ]]; then
        die "--soc-list must select at least one CANN SoC ID."
    fi
    for requested_soc in "${requested_socs[@]}"; do
        requested_soc="${requested_soc//[[:space:]]/}"
        case "${requested_soc}" in
            ascend910b|ascend910_93|ascend950)
                ;;
            *)
                die "Unsupported SoC '${requested_soc}' in --soc-list=${soc_list}; " \
                    "use ascend910b, ascend910_93, or ascend950."
                ;;
        esac
    done
}

function framework_union() {
    local first=$1
    local second=$2
    if [[ "${first}" == "off" ]]; then
        echo "${second}"
    elif [[ "${second}" == "off" || "${first}" == "${second}" ]]; then
        echo "${first}"
    elif [[ "${first}" == "all" || "${second}" == "all" ]]; then
        echo "all"
    else
        echo "all"
    fi
}

function source_default_cann_environment() {
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
        echo "INFO: using configured CANN environment: ${ASCEND_HOME_PATH}"
        return
    fi
    if [[ ! -f "${DEFAULT_CANN_SET_ENV}" ]]; then
        echo "WARNING: CANN environment is not configured and the default script is unavailable: " \
             "${DEFAULT_CANN_SET_ENV}" >&2
        return
    fi
    # shellcheck source=/dev/null
    set +e +u
    source "${DEFAULT_CANN_SET_ENV}"
    local source_status=$?
    set -e -u
    if [[ ${source_status} -ne 0 ]]; then
        echo "WARNING: failed to source the default CANN environment: ${DEFAULT_CANN_SET_ENV}" >&2
        return
    fi
    echo "INFO: sourced default CANN environment: ${DEFAULT_CANN_SET_ENV}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --multicore=*)
            MULTICORE_VALUE=$(normalize_framework "${1#*=}")
            shift
            ;;
        --multicore)
            require_value "$1" "${2:-}"
            MULTICORE_VALUE=$(normalize_framework "$2")
            shift 2
            ;;
        --shmem=*)
            SHMEM_VALUE=$(normalize_framework "${1#*=}")
            shift
            ;;
        --shmem)
            require_value "$1" "${2:-}"
            SHMEM_VALUE=$(normalize_framework "$2")
            shift 2
            ;;
        --custom-ops=*)
            CUSTOM_OPS_VALUE=$(normalize_on_off "${1#*=}")
            shift
            ;;
        --custom-ops)
            require_value "$1" "${2:-}"
            CUSTOM_OPS_VALUE=$(normalize_on_off "$2")
            shift 2
            ;;
        --strict=*)
            STRICT_VALUE=$(normalize_on_off "${1#*=}")
            shift
            ;;
        --strict)
            require_value "$1" "${2:-}"
            STRICT_VALUE=$(normalize_on_off "$2")
            shift 2
            ;;
        --soc-list=*)
            require_value "--soc-list" "${1#*=}"
            SOC_LIST_VALUE="${1#*=}"
            shift
            ;;
        --soc-list)
            require_value "$1" "${2:-}"
            SOC_LIST_VALUE="$2"
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
            die "Unknown option '$1'."
            ;;
    esac
done

if ! [[ "${NATIVE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    die "--jobs must be a positive integer, got '${NATIVE_JOBS}'."
fi
validate_soc_list "${SOC_LIST_VALUE}"

RESOLVED_SHMEM_VALUE=$(framework_union "${SHMEM_VALUE}" "${MULTICORE_VALUE}")
if [[ "${RESOLVED_SHMEM_VALUE}" != "off" || "${MULTICORE_VALUE}" != "off" || \
      "${CUSTOM_OPS_VALUE}" != "off" ]]; then
    source_default_cann_environment
fi
cd "${PROJECT_ROOT}"
rm -rf "${PAYLOAD_ROOT}"
mkdir -p "${PAYLOAD_ROOT}" "${LOG_ROOT}"

echo "Build configuration:"
printf '  %-24s %s\n' "multicore" "${MULTICORE_VALUE}"
printf '  %-24s %s\n' "symmetric_memory" "${SHMEM_VALUE}"
printf '  %-24s %s\n' "resolved_shmem" "${RESOLVED_SHMEM_VALUE}"
printf '  %-24s %s\n' "custom_ops" "${CUSTOM_OPS_VALUE}"
printf '  %-24s %s\n' "soc_list" "${SOC_LIST_VALUE}"
printf '  %-24s %s\n' "strict" "${STRICT_VALUE}"
printf '  %-24s %s\n' "jobs" "${NATIVE_JOBS}"
printf '  %-24s %s\n' "clean" "${CLEAN}"
if [[ "${RESOLVED_SHMEM_VALUE}" != "${SHMEM_VALUE}" ]]; then
    echo "INFO: symmetric_memory=${RESOLVED_SHMEM_VALUE} is required by multicore=${MULTICORE_VALUE}."
fi

function component_command() {
    local component=$1
    local -a command=()
    case "${component}" in
        symmetric_memory)
            command=(bash scripts/build_symmetric_memory.sh
                --framework "${RESOLVED_SHMEM_VALUE}"
                --soc-list "${SOC_LIST_VALUE}"
                --jobs "${NATIVE_JOBS}")
            ;;
        multicore)
            command=(bash scripts/build_multicore.sh
                --framework "${MULTICORE_VALUE}"
                --soc-list "${SOC_LIST_VALUE}"
                --jobs "${NATIVE_JOBS}")
            ;;
        custom_ops)
            command=(bash scripts/build_custom_ops.sh
                --framework mindspore
                --jobs "${NATIVE_JOBS}")
            ;;
        *)
            die "Unknown native component '${component}'."
            ;;
    esac
    if [[ "${CLEAN}" == "on" ]]; then
        command+=(--clean)
    fi
    "${command[@]}"
}

declare -a FAILED_COMPONENTS=()
function remove_component_payload() {
    local component=$1
    case "${component}" in
        symmetric_memory)
            rm -rf "${PAYLOAD_ROOT}/core/symmetric_memory"
            ;;
        multicore)
            rm -rf "${PAYLOAD_ROOT}/core/multicore/lib"
            ;;
        custom_ops)
            rm -rf "${PAYLOAD_ROOT}/platform/mindspore/custom_ops/lib"
            ;;
        *)
            die "Unknown native component '${component}'."
            ;;
    esac
}

function run_optional_component() {
    local component=$1
    local log_file="${LOG_ROOT}/${component}.log"
    echo "[HP-NATIVE] component=${component} log=${log_file}"
    set +e
    component_command "${component}" 2>&1 | tee "${log_file}"
    local component_status=${PIPESTATUS[0]}
    set -e
    if [[ ${component_status} -eq 0 ]]; then
        echo "[HP-NATIVE] result=SUCCESS component=${component}"
        return 0
    fi
    rm -rf "${COMPONENT_ROOT:?}/${component}"
    remove_component_payload "${component}"
    echo "WARNING: [HP-NATIVE-COMPONENT-FAILED] component=${component} " \
         "exit=${component_status} log=${log_file}" >&2
    FAILED_COMPONENTS+=("${component}")
    if [[ "${STRICT_VALUE}" == "on" ]]; then
        return "${component_status}"
    fi
    return 0
}

if [[ "${RESOLVED_SHMEM_VALUE}" != "off" ]]; then
    run_optional_component symmetric_memory
fi
if [[ "${MULTICORE_VALUE}" != "off" ]]; then
    run_optional_component multicore
fi
if [[ "${CUSTOM_OPS_VALUE}" != "off" ]]; then
    run_optional_component custom_ops
fi

if [[ ${#FAILED_COMPONENTS[@]} -gt 0 ]]; then
    failed_list=$(IFS=,; echo "${FAILED_COMPONENTS[*]}")
    echo "WARNING: [HP-NATIVE-SUMMARY] optional native components failed: ${failed_list}." >&2
fi

INDEXED_HELPERS_DIR="${PROJECT_ROOT}/hyper_parallel/auto_models/components/datasets/llm"
INDEXED_HELPERS_SOURCE="${INDEXED_HELPERS_DIR}/csrc/indexed_helpers.cpp"
PYTHON_BIN=${PYTHON:-python}
CXX_BIN=${CXX:-c++}
INDEXED_HELPERS_SUFFIX=$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
INDEXED_HELPERS_OUTPUT="${INDEXED_HELPERS_DIR}/_indexed_helpers_cpp${INDEXED_HELPERS_SUFFIX}"

if ! PYBIND11_INCLUDES=$("${PYTHON_BIN}" -m pybind11 --includes 2>/dev/null); then
    die "pybind11 is required to build indexed Dataset helpers. Install it with 'pip install pybind11'."
fi
read -r -a PYBIND11_INCLUDE_ARGS <<< "${PYBIND11_INCLUDES}"

echo "Building indexed Dataset helpers: ${INDEXED_HELPERS_OUTPUT}"
"${CXX_BIN}" -O3 -Wall -shared -std=c++17 -fPIC "${PYBIND11_INCLUDE_ARGS[@]}" \
    "${INDEXED_HELPERS_SOURCE}" -o "${INDEXED_HELPERS_OUTPUT}"

export HYPER_PARALLEL_NATIVE_OUTPUT_ROOT="${PAYLOAD_ROOT}"
rm -rf "${WHEEL_OUTPUT_ROOT}"
mkdir -p "${WHEEL_OUTPUT_ROOT}"
"${PYTHON_BIN}" setup.py -q bdist_wheel --dist-dir "${WHEEL_OUTPUT_ROOT}"
mapfile -t built_wheels < <(find "${WHEEL_OUTPUT_ROOT}" -maxdepth 1 -type f -name 'hyper_parallel-*.whl' -print)
if [[ ${#built_wheels[@]} -ne 1 ]]; then
    die "Expected one wheel from this build invocation, found ${#built_wheels[@]} under ${WHEEL_OUTPUT_ROOT}."
fi
mkdir -p "${PROJECT_ROOT}/dist"
wheel_path="${PROJECT_ROOT}/dist/$(basename "${built_wheels[0]}")"
cp -a "${built_wheels[0]}" "${wheel_path}"

echo "Build completed."
echo "  PYTHONPATH payload: ${PAYLOAD_ROOT}"
echo "  wheel: ${wheel_path}"
if [[ -f "${PAYLOAD_ROOT}/core/multicore/lib/set_env.bash" ]]; then
    echo "Multicore/HyperMegaMoe requires explicit activation before starting the framework Python process:"
    echo "  PYTHONPATH: source ${PAYLOAD_ROOT}/core/multicore/lib/set_env.bash"
    echo '  wheel:      source "$(command -v hyper_parallel_multicore_set_env.bash)"'
fi
