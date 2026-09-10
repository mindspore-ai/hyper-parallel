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

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo"
DATA_DIR="${OUTPUT_DIR}/kimi_vlm_data"

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
    echo "Kimi-K2.5/K2.6 config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
if [[ ! -s "${DATA_DIR}/data.json" ]]; then
    python -m examples.training_demo.prepare_kimi_vlm_data \
        --output-dir "${DATA_DIR}" \
        --num-samples 8
fi

torchrun \
    --standalone \
    --nproc_per_node=8 \
    "${PROJECT_ROOT}/scripts/train_vl.py" \
    "${SCRIPT_DIR}/train_kimi_vlm.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_DIR}/data.json" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_kimi_vlm.log"
