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

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo/deepseek_v41"
ASSETS_PATH="${OUTPUT_DIR}/engram_validation.json"
DEFAULT_DATA_PATH="${OUTPUT_DIR}/mm_data/deepseek_v41_messages/train.jsonl"
RUN_NAME=${RUN_NAME:-vlm_tp1_ep16}

if [[ ! ${RUN_NAME} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/DeepSeek-V4.1-Flash [path/to/train.jsonl] [trainer overrides...]" >&2
    exit 1
fi

MODEL_PATH=$1
shift
DATA_PATH="${DEFAULT_DATA_PATH}"
if [[ $# -gt 0 && ${1} != --* ]]; then
    DATA_PATH=$1
    shift
fi
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
DATA_PATH=$(cd "$(dirname "${DATA_PATH}")" 2>/dev/null && pwd)/$(basename "${DATA_PATH}") || {
    echo "V4.1 JSONL data does not exist: ${DATA_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local V4.1 config/tokenizer assets are incomplete: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -s "${DATA_PATH}" ]]; then
    echo "V4.1 JSONL data is missing or empty: ${DATA_PATH}" >&2
    exit 1
fi
if ! python -c "from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config"; then
    echo "Transformers DeepSeek-V4 support is unavailable." >&2
    echo "Prepare the shell with docs/guide/trainer/current_hf_model_environment.md" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/${RUN_NAME}.success"
if [[ ! -s "${ASSETS_PATH}" ]]; then
    python -m examples.training_demo.prepare_deepseek_v41_assets \
        --model-dir "${MODEL_PATH}" \
        --output "${ASSETS_PATH}" \
        --bucket-base 4096 \
        --num-hidden-layers 4
fi

torchrun \
    --standalone \
    --nproc_per_node=16 \
    "${PROJECT_ROOT}/scripts/train_vl.py" \
    "${SCRIPT_DIR}/train_deepseek_v41_vlm_online.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --model.engram_assets_path="${ASSETS_PATH}" \
    --dataset.data_transform.config_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${RUN_NAME}.log"

touch "${OUTPUT_DIR}/${RUN_NAME}.success"
