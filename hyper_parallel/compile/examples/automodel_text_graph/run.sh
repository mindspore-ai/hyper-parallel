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

# GraphTextTrainer text demo launcher — default fixed-shape TP2 + FSDP2 case.
#
# Usage:
#   bash run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_PARALLEL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${HYPER_PARALLEL_ROOT}/.." && pwd)"

CONFIG="${SCRIPT_DIR}/train_lm_graph_fixed_tp2_fsdp2.yaml"
NPROC_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29683
LABEL="train_lm_graph_fixed_tp2_fsdp2"
OUTPUT_DIR="${SCRIPT_DIR}/output"
TRAIN_ENTRY="${REPO_ROOT}/scripts/train_lm.py"

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

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --tee=3 \
    --local-ranks-filter=0 \
    "${TRAIN_ENTRY}" "${CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/run_${LABEL}.log"

echo "=========================================================="
echo "Done"
echo "log: ${OUTPUT_DIR}/run_${LABEL}.log"
echo "=========================================================="
