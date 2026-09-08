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

# TP + FSDP graph-mode demo — 4 cards (TP=2, DP/FSDP=2).
#
# Usage:
#   bash run.sh            # NPU (hccl) if available, else gloo on CPU
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "TP=2  DP/FSDP=2  (world=4) graph-mode demo"
echo "=========================================="

LABEL=${LABEL:-data}
OUTPUT_DIR="./graphtraier_output"
mkdir -p "${OUTPUT_DIR}"


torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --tee=3 \
    --local-ranks-filter=0 \
    train.py --config config.yaml \
    2>&1 | tee "${OUTPUT_DIR}/run_${LABEL}.log"

echo "=========================================="
echo "Done"
echo "=========================================="
