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

function die() {
    echo "ERROR: $*" >&2
    exit 2
}

if [[ $# -eq 1 && ( "$1" == -h || "$1" == --help ) ]]; then
    echo "Usage: bash build.sh [-h|--help]"
    echo "Build the indexed Dataset C++ helper and package HyperParallel."
    exit 0
fi
[[ $# -eq 0 ]] || die "Unknown option '$1'. Run bash build.sh --help."
PYTHON_BIN=$(command -v python) || die "Python is unavailable in the active PATH."
cd "${PROJECT_ROOT}"
mkdir -p "${PAYLOAD_STAGING_ROOT}" "${NATIVE_ROOT}/logs"
trap 'rm -rf "${PAYLOAD_STAGING_ROOT}"' EXIT
bash hyper_parallel/data/indexed/build.sh --stage-only 2>&1 | tee "${NATIVE_ROOT}/logs/indexed.log"
cp -a "${NATIVE_ROOT}/components/indexed/hyper_parallel/." "${PAYLOAD_STAGING_ROOT}/"

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
