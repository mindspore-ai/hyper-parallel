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

set -e

cd "$(dirname "$0")/../.."

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

LABEL=${LABEL:-data}
OUTPUT_DIR="${PWD}/output"
export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 8,9,10,11,12,13,14,15 # 

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export PYTHONPATH=../../hyper-parallel:$PYTHONPATH

mkdir -p "${OUTPUT_DIR}"

python -m examples.training_demo.prepare_model \
    examples/training_demo/train.yaml \
    "$@"

torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_id=training_demo \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.training_demo.train_text \
    examples/training_demo/train.yaml \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${LABEL}.log"

# $(date +%Y%m%d_%H%M%S)
