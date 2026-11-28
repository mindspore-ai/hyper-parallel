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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-${SCRIPT_DIR}/train_qwen3_graph_tp2_fsdp2.yaml}"
OUTPUT_LOG="${OUTPUT_LOG:-./output/qwen3_graph/train.log}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-HOST}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"

resource_args=()
if [[ -n "${MODEL_DIR:-}" ]]; then
    resource_args+=("--model.pretrained_model_name_or_path=${MODEL_DIR}")
fi
if [[ -n "${DATA_PATH:-}" ]]; then
    resource_args+=("--dataset.data_path=${DATA_PATH}")
fi

mkdir -p "$(dirname "${OUTPUT_LOG}")"
python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE:-4}" \
    --master_port="${MASTER_PORT:-29683}" \
    --tee=3 --local-ranks-filter=0 \
    scripts/train_lm.py "${CONFIG}" "${resource_args[@]}" "$@" 2>&1 | tee "${OUTPUT_LOG}"
