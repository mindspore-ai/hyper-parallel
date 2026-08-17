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
PROJECT_ROOT=$(pwd)

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 8,9,10,11,12,13,14,15 # 
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

# log and data
LABEL=${LABEL:-data}
OUTPUT_DIR="./output"
DEMO_ROOT="${PROJECT_ROOT}/outputs/training_demo/tiny_llama_wikitext2"
TOKENIZER_DIR="${DEMO_ROOT}/tokenizer"
DATA_OUTPUT_PREFIX="${DEMO_ROOT}/data/wikitext2_real"
DATA_PREFIX="${DATA_OUTPUT_PREFIX}_text_document"
DATA_WORKERS=${DATA_WORKERS:-8}
TOKENIZER_SOURCE=${TOKENIZER_SOURCE:-Qwen/Qwen3-0.6B}
mkdir -p "${OUTPUT_DIR}"

# Download the tokenizer only, then create a tiny random Llama checkpoint. Existing files are reused.
python -m examples.training_demo.prepare_model \
    --output-dir "${TOKENIZER_DIR}" \
    --tokenizer-source "${TOKENIZER_SOURCE}"

if [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
    echo "Missing demo tokenizer: ${TOKENIZER_DIR}/tokenizer.json" >&2
    exit 1
fi

# Download and prepare indexed WikiText-2 only when either output file is missing.
if [[ ! -f "${DATA_PREFIX}.bin" || ! -f "${DATA_PREFIX}.idx" ]]; then
    mkdir -p "${DEMO_ROOT}/data" "${DEMO_ROOT}/raw"
    python -m hyper_models.components.datasets.tools.huggingface_offline \
        --dataset Salesforce/wikitext \
        --dataset-subset wikitext-2-raw-v1 \
        --dataset-split train \
        --download-dir "${DEMO_ROOT}/raw" \
        --json-keys text \
        --output-prefix "${DATA_OUTPUT_PREFIX}" \
        --tokenizer "${TOKENIZER_DIR}" \
        --workers "${DATA_WORKERS}" \
        --append-eod \
        --pad-to-seq-len 64
fi

torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_id=training_demo \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.training_demo.train_text \
    examples/training_demo/train_idx_dataset.yaml \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${LABEL}.log"

# $(date +%Y%m%d_%H%M%S)
