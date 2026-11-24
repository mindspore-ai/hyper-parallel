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
#
# Launch train_parallel_offline.yaml on one node with 8 NPUs:
#   bash run_parallel_offline.sh /path/to/DeepSeek-V3.2-8layer-crop \
#       /path/to/dataset_text_document [trainer overrides...]
# The second argument is the Indexed Dataset prefix, without .bin/.idx.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/deepseek_v32}

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/DeepSeek-V3.2-8layer-crop /path/to/dataset_text_document [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: $1" >&2
    exit 1
}
DATA_DIR=$(cd "$(dirname "$2")" 2>/dev/null && pwd) || {
    echo "Dataset directory does not exist: $(dirname "$2")" >&2
    exit 1
}
DATA_PATH=${DATA_DIR}/$(basename "$2")
shift 2
if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "DeepSeek-V3.2 config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
if [[ ! -s "${DATA_PATH}.bin" || ! -s "${DATA_PATH}.idx" ]]; then
    echo "Indexed Dataset is missing: ${DATA_PATH}.bin and ${DATA_PATH}.idx" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
torchrun \
    --standalone \
    --nproc_per_node=8 \
    scripts/train_lm.py \
    "${SCRIPT_DIR}/train_parallel_offline.yaml" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --dataset.data_config.data_cache_path="${OUTPUT_DIR}/data_cache" \
    --checkpoint.checkpoint_dir="${OUTPUT_DIR}/checkpoints" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_parallel_offline.log"
