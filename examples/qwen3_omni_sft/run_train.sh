#!/usr/bin/env bash
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

# Qwen3-Omni-30B-A3B ScienceQA SFT on 8 x Ascend NPU (single node).
# Runs the thinker-only recipe in qwen3_omni_weights_48.yaml through
# scripts/train_vl.py; trainer overrides pass through as extra args.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# Local or offline resources: set these instead of editing the yaml.
MODEL_PATH=${MODEL_PATH:-}
DATA_PATH=${DATA_PATH:-}
CKPT_DIR=${CKPT_DIR:-./outputs/qwen3_omni/checkpoints}
N_EPOCHS=${N_EPOCHS:-2}

OVERRIDES=()
if [ -n "${MODEL_PATH}" ]; then
    OVERRIDES+=(--model.pretrained_model_name_or_path="${MODEL_PATH}"
                --dataset.model_assets.pretrained_model_name_or_path="${MODEL_PATH}")
fi
if [ -n "${DATA_PATH}" ]; then
    OVERRIDES+=(--dataset.data_path="${DATA_PATH}")
fi

# steps per epoch: drop_last floor(2048 / global_batch_size 8) = 256
OVERRIDES+=(--training.train_iters=$((256 * N_EPOCHS))
            --checkpoint.save_ckpt=true
            --checkpoint.checkpoint_dir="${CKPT_DIR}"
            --checkpoint.save_steps=0
            --checkpoint.save_epochs=1
            --checkpoint.save_consolidated=final)

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

cd "${PROJECT_ROOT}"
torchrun \
    --nproc_per_node=8 \
    --rdzv_id=qwen3_omni_sft_1node_8dies \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}" \
    --module scripts.train_vl \
    "${SCRIPT_DIR}/qwen3_omni_weights_48.yaml" \
    "${OVERRIDES[@]}" \
    "$@"
