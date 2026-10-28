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

# Launch train_online.yaml on 32 nodes with 8 NPUs each after sourcing CANN.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous node address}"
: "${NODE_RANK:?Set NODE_RANK to a unique node index from 0 to 31}"
if [[ ! "${NODE_RANK}" =~ ^([0-9]|[12][0-9]|3[01])$ ]]; then
    echo "NODE_RANK must be an integer from 0 to 31" >&2
    exit 1
fi
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-qwen3_8_32nodes_256dies}

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

cd "${PROJECT_ROOT}"
# CP=4 leaves 64 DP groups; batch 64 runs one micro-batch per optimizer step.
torchrun \
    --nnodes=32 \
    --nproc_per_node=8 \
    --node_rank="${NODE_RANK}" \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=static \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module scripts.train_lm \
    "${SCRIPT_DIR}/train_online.yaml" \
    --fsdp_config.dp_shard_size=256 \
    --accelerator.cp_size=4 \
    --training.global_batch_size=64 \
    "$@"
