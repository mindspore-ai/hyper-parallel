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
source "${SCRIPT_DIR}/run_training.bash"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${DATA_ROOT}/raw" "${DATA_ROOT}/indexed"

# Download the shared WikiText source when it is not available locally.
if [[ ! -s "${RAW_DATA_PATH}" ]]; then
    python -m hyper_models.components.datasets.tools.huggingface_offline \
        --dataset Salesforce/wikitext \
        --dataset-subset wikitext-2-raw-v1 \
        --dataset-split train \
        --download-dir "${DATA_ROOT}/raw" \
        --json-keys text \
        --output-prefix "${INDEXED_OUTPUT_PREFIX}" \
        --tokenizer "${MODEL_SOURCE}" \
        --workers "${DATA_WORKERS}" \
        --append-eod true
fi

if [[ ! -s "${RAW_DATA_PATH}" ]]; then
    echo "Required training data file is missing or empty: ${RAW_DATA_PATH}" >&2
    exit 1
fi

# Run the fixed-batch Online baseline.
MASTER_PORT=${MASTER_PORT:-29501}
torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_id=training_demo_base \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_base.yaml" \
    --model.pretrained_model_name_or_path="${MODEL_SOURCE}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_SOURCE}" \
    --dataset.data_path="${RAW_DATA_PATH}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_base.log"
