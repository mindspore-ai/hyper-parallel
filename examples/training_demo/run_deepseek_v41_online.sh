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
DATA_PATH="${OUTPUT_DIR}/online_4k.jsonl"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/DeepSeek-V4.1-Flash tp1|tp2|cp2 [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1
MODE=$2
shift 2
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local V4.1 config/tokenizer assets are incomplete: ${MODEL_PATH}" >&2
    exit 1
fi
if ! python -c "from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config"; then
    echo "Transformers DeepSeek-V4 support is unavailable." >&2
    echo "Prepare the shell with docs/guide/trainer/current_hf_model_environment.md" >&2
    exit 1
fi
case "${MODE}" in
    tp1)
        TOPOLOGY_OVERRIDES=(
            --accelerator.tp_size=1
            --accelerator.cp_size=1
            --fsdp_config.dp_shard_size=16
            --training.global_batch_size=16
        )
        ;;
    tp2)
        if [[ ! -f "${OUTPUT_DIR}/tp1.success" ]]; then
            echo "Run a successful tp1 smoke before tp2; ${OUTPUT_DIR}/tp1.success is missing" >&2
            exit 1
        fi
        TOPOLOGY_OVERRIDES=(
            --accelerator.tp_size=2
            --accelerator.cp_size=1
            --accelerator.sequence_parallel=true
            --fsdp_config.dp_shard_size=8
            --training.global_batch_size=8
        )
        ;;
    cp2)
        TOPOLOGY_OVERRIDES=(
            --accelerator.tp_size=1
            --accelerator.cp_size=2
            --accelerator.sequence_parallel=false
            --fsdp_config.dp_shard_size=16
            --training.global_batch_size=8
        )
        ;;
    *)
        echo "Mode must be tp1, tp2, or cp2, got: ${MODE}" >&2
        exit 1
        ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
if [[ ! -s "${ASSETS_PATH}" ]]; then
    python -m examples.training_demo.prepare_deepseek_v41_assets \
        --model-dir "${MODEL_PATH}" \
        --output "${ASSETS_PATH}" \
        --bucket-base 4096 \
        --num-hidden-layers 4
fi
if [[ ! -s "${DATA_PATH}" ]]; then
    python -m examples.training_demo.prepare_deepseek_v41_online_data \
        --output "${DATA_PATH}" \
        --num-samples 128 \
        --sequence-length 4096
fi

torchrun \
    --standalone \
    --nproc_per_node=16 \
    --module examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_deepseek_v41_online.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --model.engram_assets_path="${ASSETS_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    "${TOPOLOGY_OVERRIDES[@]}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${MODE}.log"

touch "${OUTPUT_DIR}/${MODE}.success"
