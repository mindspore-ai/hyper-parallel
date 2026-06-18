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
PROJECT_ROOT="${SCRIPT_DIR}"

MULTICORE_VALUE="mindspore"
SHMEM_VALUE="all"
CUSTOM_OPS_VALUE="on"
STRICT_VALUE="on"

function show_help() {
    cat <<EOF
Usage:
  ./build.sh [OPTIONS]

Options:
  --multicore VALUE          Build multicore: off, mindspore/ms, torch/pytorch, all/both.
                             Default: mindspore.
  --shmem VALUE              Build symmetric memory: off, mindspore/ms, torch/pytorch, all/both.
                             Default: all.
  --custom-ops VALUE         Build custom ops: on or off. Default: on.
  --strict VALUE             Fail when an optional native build step fails. Default: on.
                             Set to off to keep setup.py's warning-and-continue behavior.
  -h, --help                 Show this help message.

Examples:
  ./build.sh
  ./build.sh --multicore all --shmem all --custom-ops on
  ./build.sh --multicore torch --shmem torch --strict off
  ./build.sh --multicore off --shmem off --custom-ops off
EOF
}

function die() {
    echo "ERROR: $*" >&2
    echo "Run './build.sh --help' for usage." >&2
    exit 1
}

function normalize_lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

function require_value() {
    local option_name=$1
    local option_value=${2:-}
    if [[ -z "${option_value}" || "${option_value}" == --* ]]; then
        die "${option_name} requires a value."
    fi
}

function validate_multicore_value() {
    local value
    value=$(normalize_lower "$1")
    case "${value}" in
        off|mindspore|ms|torch|pytorch|all|both)
            echo "${value}"
            ;;
        *)
            die "Unsupported --multicore value '${1}'. Use off, mindspore/ms, torch/pytorch, or all/both."
            ;;
    esac
}

function validate_shmem_value() {
    local value
    value=$(normalize_lower "$1")
    case "${value}" in
        off|mindspore|ms|torch|pytorch|all|both)
            echo "${value}"
            ;;
        *)
            die "Unsupported --shmem value '${1}'. Use off, mindspore/ms, torch/pytorch, or all/both."
            ;;
    esac
}

function validate_custom_ops_value() {
    local value
    value=$(normalize_lower "$1")
    case "${value}" in
        on)
            echo "on"
            ;;
        off)
            echo "off"
            ;;
        *)
            die "Unsupported --custom-ops value '${1}'. Use on or off."
            ;;
    esac
}

function validate_strict_value() {
    local value
    value=$(normalize_lower "$1")
    case "${value}" in
        on)
            echo "on"
            ;;
        off)
            echo "off"
            ;;
        *)
            die "Unsupported --strict value '${1}'. Use on or off."
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --multicore=*)
            MULTICORE_VALUE=$(validate_multicore_value "${1#*=}")
            shift
            ;;
        --multicore)
            require_value "$1" "${2:-}"
            MULTICORE_VALUE=$(validate_multicore_value "$2")
            shift 2
            ;;
        --shmem=*)
            SHMEM_VALUE=$(validate_shmem_value "${1#*=}")
            shift
            ;;
        --shmem)
            require_value "$1" "${2:-}"
            SHMEM_VALUE=$(validate_shmem_value "$2")
            shift 2
            ;;
        --custom-ops=*)
            CUSTOM_OPS_VALUE=$(validate_custom_ops_value "${1#*=}")
            shift
            ;;
        --custom-ops)
            require_value "$1" "${2:-}"
            CUSTOM_OPS_VALUE=$(validate_custom_ops_value "$2")
            shift 2
            ;;
        --strict=*)
            STRICT_VALUE=$(validate_strict_value "${1#*=}")
            shift
            ;;
        --strict)
            require_value "$1" "${2:-}"
            STRICT_VALUE=$(validate_strict_value "$2")
            shift 2
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

export BUILD_MULTICORE_EXTENSION="${MULTICORE_VALUE}"
export BUILD_SHMEM_EXTENSION="${SHMEM_VALUE}"
export BUILD_CUSTOM_OPS_EXTENSION="${CUSTOM_OPS_VALUE}"
export HYPER_PARALLEL_BUILD_STRICT="${STRICT_VALUE}"

cd "${PROJECT_ROOT}"

echo "Build configuration:"
printf '  %-32s %s\n' "BUILD_MULTICORE_EXTENSION" "${BUILD_MULTICORE_EXTENSION}"
printf '  %-32s %s\n' "BUILD_SHMEM_EXTENSION" "${BUILD_SHMEM_EXTENSION}"
printf '  %-32s %s\n' "BUILD_CUSTOM_OPS_EXTENSION" "${BUILD_CUSTOM_OPS_EXTENSION}"
printf '  %-32s %s\n' "HYPER_PARALLEL_BUILD_STRICT" "${HYPER_PARALLEL_BUILD_STRICT}"

python setup.py bdist_wheel
