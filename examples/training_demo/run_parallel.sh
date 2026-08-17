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

cd "$(dirname "$0")/../.."

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29520}
DATA_MODE=${DATA_MODE:-online}
DATA_DIR=${DATA_DIR:-"${PWD}/outputs/training_demo/parallel_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PWD}/output"}

case "${DATA_MODE}" in
    online)
        CONFIG_PATH=examples/training_demo/train_parallel_online.yaml
        DATA_PATH="${DATA_DIR}/parallel_online.jsonl"
        ;;
    offline)
        CONFIG_PATH=examples/training_demo/train_parallel_offline.yaml
        DATA_PATH="${DATA_DIR}/parallel_offline_text_document"
        ;;
    *)
        echo "DATA_MODE must be 'online' or 'offline', got '${DATA_MODE}'" >&2
        exit 2
        ;;
esac

export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
python -m examples.tp_cp_ep_demo.prepare_model "${CONFIG_PATH}" "$@"
python -m examples.training_demo.prepare_parallel_data --output-dir "${DATA_DIR}"

torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_id="training_demo_parallel_${DATA_MODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.training_demo.train_text \
    "${CONFIG_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_parallel_${DATA_MODE}.log"
