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

# Keep generated assets and run outputs under the repository output directory.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/output/training_demo/deepseek_v41"}
ASSETS_PATH=${ASSETS_PATH:-"${OUTPUT_DIR}/artifacts/engram_depth_preserving_d4.json"}
DEFAULT_DATA_PATH="${OUTPUT_DIR}/online_4k.jsonl"
RUN_NAME=${RUN_NAME:-text_tp1_cp4_ep8_fsdp8}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}

if [[ ! ${RUN_NAME} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/DeepSeek-V4.1-Flash [path/to/train.jsonl] [trainer overrides...]" >&2
    exit 1
fi

# Resolve input paths before switching to the repository root.
MODEL_PATH=$1
shift
DATA_PATH="${DEFAULT_DATA_PATH}"
if [[ $# -gt 0 && ${1} != --* ]]; then
    DATA_PATH=$1
    if [[ ! -s "${DATA_PATH}" ]]; then
        echo "Text JSONL data is missing or empty: ${DATA_PATH}" >&2
        exit 1
    fi
    shift
fi
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ "${DATA_PATH}" != /* ]]; then
    DATA_PATH="${PWD}/${DATA_PATH}"
fi
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local V4.1 config/tokenizer assets are incomplete: ${MODEL_PATH}" >&2
    exit 1
fi
# Load native operators from the selected training environment.
cd "${PROJECT_ROOT}"
PYTHON_BIN=$(command -v "${PYTHON_BIN}")
PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PATH
# shellcheck source=/dev/null
source "$(command -v hyper_multicore_set_env.bash)"
# shellcheck source=/dev/null
source "${OPS_TRANSFORMER_ROOT}/install/vendors/custom_transformer/bin/set_env.bash"

mkdir -p "${OUTPUT_DIR}"
# Clear the previous success marker before starting a new run.
rm -f "${OUTPUT_DIR}/${RUN_NAME}.success"
# Generate Engram assets only when the cached file is missing.
if [[ ! -s "${ASSETS_PATH}" ]]; then
    "${PYTHON_BIN}" -m hyper_parallel.models.deepseek_v41.tools.prepare_deepseek_v41_assets \
        --model-dir "${MODEL_PATH}" \
        --output "${ASSETS_PATH}" \
        --bucket-base 4096 \
        --parameter-divisor 4
fi

# Generate deterministic text samples when the default dataset is absent.
if [[ ! -s "${DATA_PATH}" ]]; then
    "${PYTHON_BIN}" -m hyper_parallel.models.deepseek_v41.tools.prepare_deepseek_v41_online_data \
        --output "${DATA_PATH}" \
        --num-samples 128 \
        --sequence-length 4096
fi

# Launch with the YAML settings; trailing Trainer arguments override its defaults.
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --module examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_deepseek_v41_online.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --model.engram_assets_path="${ASSETS_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --checkpoint.checkpoint_dir="${OUTPUT_DIR}/artifacts/text_checkpoints" \
    --profiler.output_path="${OUTPUT_DIR}/text_profiler" \
    --memory.save_path="${OUTPUT_DIR}/text_memory_snapshot" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${RUN_NAME}.log"

# Mark success only after all training processes finish successfully.
touch "${OUTPUT_DIR}/${RUN_NAME}.success"
