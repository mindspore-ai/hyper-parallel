#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANTS=(
    sync_colossal
    sync_load_balance
    sync_ulysses
    sync_hybrid
    async_colossal
    async_ulysses
    async_hybrid
)

SELECTED_VARIANT=${1:-all}
if [[ $# -gt 0 ]]; then
    shift
fi

variant_exists=false
for variant in "${VARIANTS[@]}"; do
    if [[ "${SELECTED_VARIANT}" == "${variant}" ]]; then
        variant_exists=true
        break
    fi
done
if [[ "${SELECTED_VARIANT}" != "all" && "${variant_exists}" != "true" ]]; then
    echo "Unknown CP wrapper variant: ${SELECTED_VARIANT}" >&2
    echo "Available variants: all ${VARIANTS[*]}" >&2
    exit 1
fi

run_variant() {
    local variant=$1
    local index=$2
    shift 2

    local config="examples/training_demo/cp_configs/${variant}.yaml"
    local master_port=$((29520 + index))
    local hccl_base_port=$((11500 + index * 20))

    echo "Running ${variant} with ${config}"
    CONFIG="${config}" \
    LABEL="cp_${variant}" \
    MASTER_PORT="${master_port}" \
    HCCL_IF_BASE_PORT="${hccl_base_port}" \
        bash examples/training_demo/run.sh 8 "$@"
}

for index in "${!VARIANTS[@]}"; do
    variant=${VARIANTS[${index}]}
    if [[ "${SELECTED_VARIANT}" == "all" || "${SELECTED_VARIANT}" == "${variant}" ]]; then
        run_variant "${variant}" "${index}" "$@"
    fi
done
