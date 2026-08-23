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

PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/output"}

# dataset root path, default to xx/dataset
DATASET_ROOT=${DATASET_ROOT:-./dataset}
DATASET_ROOT=${DATASET_ROOT%/}

DEMO_ROOT=${DEMO_ROOT:-"${DATASET_ROOT}/qwen3-moe-tiny"}
MODEL_SOURCE=${MODEL_SOURCE:-"${DEMO_ROOT}/model"}

DATA_ROOT=${DATA_ROOT:-"${DEMO_ROOT}/data"}
RAW_DATA_PATH=${RAW_DATA_PATH:-"${DATA_ROOT}/raw/wikitext-2-raw-v1-train.jsonl"}
CHAT_DATA_PATH=${CHAT_DATA_PATH:-"${DATA_ROOT}/raw/validation_chat.jsonl"}
INDEXED_OUTPUT_PREFIX=${INDEXED_OUTPUT_PREFIX:-"${DATA_ROOT}/indexed/wikitext2_parallel"}
INDEXED_DATA_PATH="${INDEXED_OUTPUT_PREFIX}_text_document"
DATA_WORKERS=${DATA_WORKERS:-8}

export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export SSL_CERT_FILE=${SSL_CERT_FILE:-/home/ma-user/.codex/ca-bundle-with-huawei.pem}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
