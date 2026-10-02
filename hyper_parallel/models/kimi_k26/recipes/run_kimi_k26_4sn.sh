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
#
# Launch the Kimi-K2.6 4-supernode recipe (train.yaml in this directory).
#
# train.yaml is dimensioned for 512 Ascend NPU cards: dp_shard 512 = edp 4 x
# ep 128, so the default is 64 nodes x 8 cards. The script refuses any other
# world size unless ALLOW_OTHER_WORLD_SIZE=1, because a mismatched world size
# changes the parallel plan silently instead of failing.
#
# Run it on every node, with NODE_RANK 0..NNODES-1:
#   NNODES=64 NODE_RANK=0 MASTER_ADDR=<rank0-ip> MASTER_PORT=6100 \
#       bash run_kimi_k26_4sn.sh /path/to/Kimi-K2.6 [trainer overrides...]
#
# Environment:
#   NNODES                  number of nodes (default 64)
#   NPROC_PER_NODE          cards per node (default 8)
#   NODE_RANK               rank of this node (default 0)
#   MASTER_ADDR/MASTER_PORT rendezvous endpoint (default 127.0.0.1:6100)
#   OUTPUT_DIR              logs and demo data (default <repo>/output/kimi_k26)
#   DATA_DIR                dataset directory (default <OUTPUT_DIR>/kimi_k26_demo_data)
#   RUN_NAME                log prefix (default kimi_k26_4sn)
#   PYTHON_BIN              interpreter used for data preparation (default python)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/kimi_k26}
DATA_DIR=${DATA_DIR:-${OUTPUT_DIR}/kimi_k26_demo_data}
RUN_NAME=${RUN_NAME:-kimi_k26_4sn}
PYTHON_BIN=${PYTHON_BIN:-python}

NNODES=${NNODES:-64}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6100}
ALLOW_OTHER_WORLD_SIZE=${ALLOW_OTHER_WORLD_SIZE:-0}

if [[ ! ${RUN_NAME} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/Kimi-K2.6 [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1
shift
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "Kimi-K2.6 config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

if [[ ${NODE_RANK} -ge ${NNODES} ]]; then
    echo "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}" >&2
    exit 1
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [[ ${WORLD_SIZE} -ne 512 && ${ALLOW_OTHER_WORLD_SIZE} != 1 ]]; then
    echo "train.yaml is dimensioned for 512 cards (4 supernodes), got" >&2
    echo "NNODES * NPROC_PER_NODE = ${WORLD_SIZE}. Set ALLOW_OTHER_WORLD_SIZE=1 and" >&2
    echo "override the accelerator/fsdp degrees to run another shape." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"
if [[ ! -s "${DATA_DIR}/data.json" ]]; then
    "${PYTHON_BIN}" -m hyper_parallel.models.kimi_k26.recipes.prepare_kimi_vlm_data \
        --output-dir "${DATA_DIR}" \
        --num-samples 1024
fi

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/train_vl.py" \
    "${SCRIPT_DIR}/train.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --model.local_files_only=true \
    --dataset.model_assets.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.local_files_only=true \
    --dataset.data_path="${DATA_DIR}/data.json" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"
