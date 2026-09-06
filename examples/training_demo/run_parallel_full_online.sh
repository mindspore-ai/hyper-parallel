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
DATA_ROOT="${OUTPUT_DIR}/data"
CONFIG_FILE="${PROJECT_ROOT}/hyper_parallel/models/qwen3_moe/recipes/train.yaml"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/Qwen3-30B-A3B [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1
shift

MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen3-30B-A3B config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${DATA_ROOT}"
if [[ ! -s "${DATA_ROOT}/parallel_online.jsonl" ]]; then
    python -m examples.training_demo.prepare_parallel_data \
        --output-dir "${DATA_ROOT}" \
        --num-samples 128 \
        --seq-length 128
fi

torchrun \
    --standalone \
    --nproc_per_node=8 \
    --module examples.training_demo.train_text \
    "${CONFIG_FILE}" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --model.local_files_only=true \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.local_files_only=true \
    --dataset.data_path="${DATA_ROOT}/parallel_online.jsonl" \
    --dataset.data_config.hf_dataset_name=null \
    --dataset.data_transform.max_seq_len=128 \
    --dataloader.min_buffered_samples=8 \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_parallel_full_online.log"
