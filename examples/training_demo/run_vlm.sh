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
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo/vlm"
PLOG_DIR="${OUTPUT_DIR}/plog"

ASSET_ROOT=XX/training_demo/vlm
DEFAULT_MODEL_PATH="${ASSET_ROOT}/model/Qwen2-VL-tiny-random"
DEFAULT_DATA_PATH="${ASSET_ROOT}/data/raw/synthetic.json"
RUN_NAME=${RUN_NAME:-vlm_smoke}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

if [[ ! ${RUN_NAME} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 1
fi
if [[ ! ${NPROC_PER_NODE} =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi

MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_PATH}}
DATA_PATH=${DATA_PATH:-${DEFAULT_DATA_PATH}}
if [[ $# -gt 0 && ${1} != --* ]]; then
    MODEL_PATH=$1
    shift
fi
if [[ $# -gt 0 && ${1} != --* ]]; then
    DATA_PATH=$1
    shift
fi
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
DATA_PATH=$(cd "$(dirname "${DATA_PATH}")" 2>/dev/null && pwd)/$(basename "${DATA_PATH}") || {
    echo "Data parent directory does not exist: ${DATA_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local model config/tokenizer assets are incomplete: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -s "${DATA_PATH}" ]]; then
    echo "VLM data file is missing or empty: ${DATA_PATH}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${PLOG_DIR}"

export ASCEND_PROCESS_LOG_PATH="${PLOG_DIR}"
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --module scripts.train_vl \
    "${SCRIPT_DIR}/train_vlm.yaml" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --training.global_batch_size="${NPROC_PER_NODE}" \
    --fsdp_config.dp_shard_size="${NPROC_PER_NODE}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${RUN_NAME}.log"
