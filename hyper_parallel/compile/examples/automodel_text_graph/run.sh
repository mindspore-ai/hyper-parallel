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

# TextTrainer joint-graph launcher (TP2 + FSDP2 by default).
#
# Usage:
#   bash run.sh [config.yaml] [--config.key=value ...]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_PARALLEL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${HYPER_PARALLEL_ROOT}/.." && pwd)"

CONFIG="${1:-${CONFIG:-${SCRIPT_DIR}/train_lm_graph_tp2_fsdp2.yaml}}"
if [[ $# -gt 0 ]]; then shift; fi
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29683}"
LABEL="$(basename "${CONFIG}" .yaml)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
TRAIN_ENTRY="${TRAIN_ENTRY:-${REPO_ROOT}/scripts/train_lm.py}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "GraphTextTrainer TP2 + FSDP2 training launcher"
echo "=========================================================="
echo "repo       : ${REPO_ROOT}"
echo "entry      : ${TRAIN_ENTRY}"
echo "config     : ${CONFIG}"
echo "nproc      : ${NPROC_PER_NODE}"
echo "master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "output_dir : ${OUTPUT_DIR}"
echo "=========================================================="

"${PYTHON:-python}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --tee=3 \
    --local-ranks-filter=0 \
    "${TRAIN_ENTRY}" "${CONFIG}" "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${LABEL}.log"

echo "=========================================================="
echo "Done"
echo "log: ${OUTPUT_DIR}/run_${LABEL}.log"
echo "=========================================================="
