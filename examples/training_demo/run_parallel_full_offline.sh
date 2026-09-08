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
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo"
CONFIG_FILE="${SCRIPT_DIR}/train_parallel_full_offline.yaml"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/Qwen3-30B-A3B /path/to/offline_text_document [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1
DATA_PREFIX=$2
shift 2

MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen3-30B-A3B config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
DATA_PREFIX_DIR=$(cd "$(dirname "${DATA_PREFIX}")" 2>/dev/null && pwd) || {
    echo "Offline Indexed Dataset directory does not exist: $(dirname "${DATA_PREFIX}")" >&2
    exit 1
}
DATA_PREFIX="${DATA_PREFIX_DIR}/$(basename "${DATA_PREFIX}")"
if [[ ! -s "${DATA_PREFIX}.bin" || ! -s "${DATA_PREFIX}.idx" ]]; then
    echo "Offline Indexed Dataset is missing: ${DATA_PREFIX}.{bin,idx}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
torchrun \
    --standalone \
    --nproc_per_node=8 \
    --module examples.training_demo.train_text \
    "${CONFIG_FILE}" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --model.local_files_only=true \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.local_files_only=true \
    --dataset.model_assets.tokenizer.tokenizer_type=pretokenized \
    --dataset.model_assets.tokenizer.vocab_size=151936 \
    --dataset.model_assets.tokenizer.eod_token_id=151645 \
    --dataset.data_path="${DATA_PREFIX}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_parallel_full_offline.log"
