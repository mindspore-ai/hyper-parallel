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
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
NATIVE_ROOT="${PROJECT_ROOT}/build/native"
PAYLOAD_ROOT="${NATIVE_ROOT}/payload/hyper_parallel"
PAYLOAD_STAGING_ROOT="${NATIVE_ROOT}/payload-staging/full-build.$$"
MULTICORE_VALUE=on
CUSTOM_OPS_VALUE=on
STRICT_VALUE=off
SOC_LIST_VALUE="ascend910b,ascend910_93"
NATIVE_JOBS="$(nproc)"
CLEAN=off

function die() {
    echo "ERROR: $*" >&2
    exit 2
}

function show_help() {
    cat <<EOF
Usage: bash build.sh [OPTIONS]
  --multicore on|off   Build Multicore, including private SHMEM. Default: on.
  --custom-ops on|off  Build MindSpore custom ops. Default: on.
  --soc-list VALUE    Multicore targets. Default: ascend910b,ascend910_93.
  --jobs VALUE        Parallel build jobs. Default: nproc.
  --strict on|off     Stop on optional component failure. Default: off.
  --clean             Rebuild selected components without their build caches.
  -h, --help          Show this help.
EOF
}

function normalize_on_off() {
    local value
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "${value}" in
        on|off) echo "${value}" ;;
        *) die "Unsupported value '$1'; use on or off." ;;
    esac
}

while [[ $# -gt 0 ]]; do
    option=${1%%=*}
    case "${option}" in
        --multicore|--custom-ops|--strict|--soc-list|--jobs)
            if [[ "$1" == *=* ]]; then
                value=${1#*=}
                shift
            else
                [[ $# -ge 2 && "$2" != --* ]] || die "${option} requires a value."
                value=$2
                shift 2
            fi
            [[ -n "${value}" ]] || die "${option} requires a value."
            case "${option}" in
                --multicore) MULTICORE_VALUE=$(normalize_on_off "${value}") ;;
                --custom-ops) CUSTOM_OPS_VALUE=$(normalize_on_off "${value}") ;;
                --strict) STRICT_VALUE=$(normalize_on_off "${value}") ;;
                --soc-list) SOC_LIST_VALUE=${value} ;;
                --jobs) NATIVE_JOBS=${value} ;;
            esac
            ;;
        --clean) CLEAN=on; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) die "Unknown option '$1'." ;;
    esac
done
[[ "${NATIVE_JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer."
PYTHON_BIN=$(command -v python) || die "Python is unavailable in the active PATH."

if [[ "${MULTICORE_VALUE}" == on || "${CUSTOM_OPS_VALUE}" == on ]] && \
        [[ -z "${ASCEND_HOME_PATH:-}" && -f /usr/local/Ascend/cann/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    set -u
fi
cd "${PROJECT_ROOT}"
rm -rf "${PAYLOAD_STAGING_ROOT}"
mkdir -p "${PAYLOAD_STAGING_ROOT}" "${NATIVE_ROOT}/logs"
trap 'rm -rf "${PAYLOAD_STAGING_ROOT}"' EXIT

function run_component() {
    local component=$1
    local entry=$2
    shift 2
    local -a command=(bash "${entry}" --stage-only "$@")
    if [[ "${CLEAN}" == on ]]; then
        command+=(--clean)
    fi
    echo "[HP-NATIVE] component=${component}"
    set +e
    "${command[@]}" 2>&1 | tee "${NATIVE_ROOT}/logs/${component}.log"
    local result=${PIPESTATUS[0]}
    set -e
    if [[ ${result} -ne 0 ]]; then
        echo "WARNING: component=${component} failed with exit ${result}." >&2
        return "${result}"
    fi
    # Copy only a successful component; a copy failure aborts full packaging.
    cp -a "${NATIVE_ROOT}/components/${component}/hyper_parallel/." "${PAYLOAD_STAGING_ROOT}/" || exit 1
}

if [[ "${MULTICORE_VALUE}" == on ]]; then
    if run_component multicore hyper_parallel/core/multicore/build.sh \
            --soc-list "${SOC_LIST_VALUE}" --jobs "${NATIVE_JOBS}"; then
        :
    else
        result=$?
        # Component-owned option errors are user input failures, not optional build failures.
        [[ ${result} -ne 2 && ${result} -ne 10 ]] || exit "${result}"
        [[ "${STRICT_VALUE}" == off ]] || exit "${result}"
    fi
fi
if [[ "${CUSTOM_OPS_VALUE}" == on ]]; then
    if run_component custom_ops hyper_parallel/platform/mindspore/custom_ops/build.sh --jobs "${NATIVE_JOBS}"; then
        :
    else
        result=$?
        [[ "${STRICT_VALUE}" == off ]] || exit "${result}"
    fi
fi
run_component indexed hyper_parallel/data/indexed/build.sh

rm -rf "${PAYLOAD_ROOT}"
mkdir -p "$(dirname "${PAYLOAD_ROOT}")"
mv "${PAYLOAD_STAGING_ROOT}" "${PAYLOAD_ROOT}"
trap - EXIT
export HYPER_PARALLEL_NATIVE_OUTPUT_ROOT="${PAYLOAD_ROOT}"
WHEEL_OUTPUT_ROOT="${NATIVE_ROOT}/wheel-output"
rm -rf "${WHEEL_OUTPUT_ROOT}"
mkdir -p "${WHEEL_OUTPUT_ROOT}"
"${PYTHON_BIN}" setup.py -q bdist_wheel --dist-dir "${WHEEL_OUTPUT_ROOT}"
mapfile -t built_wheels < <(find "${WHEEL_OUTPUT_ROOT}" -maxdepth 1 -type f -name 'hyper_parallel-*.whl' -print)
[[ ${#built_wheels[@]} -eq 1 ]] || die "Expected one wheel under ${WHEEL_OUTPUT_ROOT}."
mkdir -p "${PROJECT_ROOT}/dist"
cp -a "${built_wheels[0]}" "${PROJECT_ROOT}/dist/"
echo "Build completed. PYTHONPATH payload: ${PAYLOAD_ROOT}"
echo "Wheel: ${PROJECT_ROOT}/dist/$(basename "${built_wheels[0]}")"
