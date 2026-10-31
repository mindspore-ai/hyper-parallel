#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo"
CONFIG_FILE="${PROJECT_ROOT}/hyper_parallel/models/xing40/recipes/train.yaml"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/indexed_text_document [trainer overrides...]" >&2
    echo "  e.g. --checkpoint.restore_from=/path/to/converted_checkpoint" >&2
    echo "       --accelerator.ep_size=8" >&2
    exit 1
fi
DATA_PREFIX=$1
shift

DATA_PREFIX_DIR=$(cd "$(dirname "${DATA_PREFIX}")" 2>/dev/null && pwd) || {
    echo "Indexed Dataset directory does not exist: $(dirname "${DATA_PREFIX}")" >&2
    exit 1
}
DATA_PREFIX="${DATA_PREFIX_DIR}/$(basename "${DATA_PREFIX}")"
if [[ ! -s "${DATA_PREFIX}.bin" || ! -s "${DATA_PREFIX}.idx" ]]; then
    echo "Indexed Dataset is missing: ${DATA_PREFIX}.{bin,idx}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
torchrun \
    --standalone \
    --nproc_per_node=8 \
    "${PROJECT_ROOT}/examples/training_demo/train_text.py" \
    "${CONFIG_FILE}" \
    --dataset.data_path="${DATA_PREFIX}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/xing40.log"
