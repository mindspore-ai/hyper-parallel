#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for details.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STAGE_ONLY=off
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage-only) STAGE_ONLY=on; shift ;;
        --clean) shift ;; # This small component is rebuilt on every invocation.
        -h|--help) echo "Usage: bash hyper_parallel/data/indexed/build.sh [--clean]"; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done
OUTPUT_ROOT="${PROJECT_ROOT}/build/native/components/indexed/hyper_parallel/data/indexed"
rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
SUFFIX=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
make -B -C "${SCRIPT_DIR}/csrc" "PYTHON=$(command -v python)" \
    "OUTPUT=${OUTPUT_ROOT}/_indexed_helpers_cpp${SUFFIX}.tmp"
mv "${OUTPUT_ROOT}/_indexed_helpers_cpp${SUFFIX}.tmp" "${OUTPUT_ROOT}/_indexed_helpers_cpp${SUFFIX}"
if [[ "${STAGE_ONLY}" == "off" ]]; then
    PAYLOAD_COMPONENT="${PROJECT_ROOT}/build/native/payload/hyper_parallel/data/indexed"
    rm -rf "${PAYLOAD_COMPONENT}"
    mkdir -p "${PAYLOAD_COMPONENT}"
    cp -a "${OUTPUT_ROOT}/." "${PAYLOAD_COMPONENT}/"
fi
