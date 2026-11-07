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
# Single-node generic HF text-training launcher.
#
#   $0 /path/to/hf-model <iters> [trainer overrides...]
#
# Site-specific pieces come from the environment so the script carries no
# internal paths (fill them from your environment profile):
#   CANN_SETENV   path to the CANN set_env.sh
#                 (default: /usr/local/Ascend/ascend-toolkit/set_env.sh)
#   PYTHON_BIN    python with torch_npu (default: python)
#   NPROC_PER_NODE  dies to use (default: 8)
#   EXTRA_PYTHONPATH  optional dependency overlay directory, prepended —
#                 keep any overlay BEFORE the interpreter's site-packages.
#
# Memory notes from the 9B bring-up: full-depth training needs activation
# checkpointing (eager attention retains O(seq^2) scores otherwise), and
# float32 initialization currently materializes the whole model per rank
# before sharding — pass --model_init_dtype=bfloat16 on 64GB dies until
# sharded init lands. A 32B registered-family bring-up then hit a harder
# wall: trainer init peaks at ~2.3x the bf16 parameter bytes on that path
# (~12B ceiling on 64GB dies), while a zero-registration 15.7B passed the
# same recipe — matrix in the sharded-init issue thread; crop layers
# (--model.num_hidden_layers) for smokes beyond the ceiling.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo/hf_text_bringup"
DATA_PATH="${OUTPUT_DIR}/online_2k.jsonl"
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CANN_SETENV=${CANN_SETENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/hf-model <train_iters> [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1
ITERS=$2
shift 2
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local config/tokenizer assets are incomplete: ${MODEL_PATH}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${CANN_SETENV}"
export HYPER_PARALLEL_PLATFORM=torch
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${EXTRA_PYTHONPATH:+${EXTRA_PYTHONPATH}:}${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
find "${PROJECT_ROOT}" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

mkdir -p "${OUTPUT_DIR}"
if [[ ! -s "${DATA_PATH}" ]]; then
    "${PYTHON_BIN}" -m examples.training_demo.hf_text_bringup.prepare_text_data \
        --output "${DATA_PATH}" --num-samples 128 --sequence-length 2048
fi

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --module examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_hf_text.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --training.train_iters="${ITERS}" "$@"
