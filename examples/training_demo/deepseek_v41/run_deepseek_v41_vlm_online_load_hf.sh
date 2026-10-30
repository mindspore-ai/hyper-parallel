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
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUN_NAME=${RUN_NAME:-vlm_load_hf_2text_2vision_tp1_ep8_fsdp8}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/runs/deepseek_v41/${RUN_NAME}"}

if [[ $# -lt 1 || ! ${RUN_NAME} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: $0 /path/to/HF-weights [path/to/train.jsonl] [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$(cd "$1" 2>/dev/null && pwd) || {
    echo "HF directory does not exist: $1" >&2
    exit 1
}
shift
DATA_PATH=""
if [[ $# -gt 0 && $1 != --* ]]; then
    DATA_PATH=$1
    shift
fi
PROCESSOR_PATH=${PROCESSOR_PATH:-"${MODEL_PATH}"}
PROCESSOR_PATH=$(cd "${PROCESSOR_PATH}" 2>/dev/null && pwd) || {
    echo "Processor assets do not exist; set PROCESSOR_PATH." >&2
    exit 1
}
for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
    if [[ ! -s "${MODEL_PATH}/${required}" ]]; then
        echo "HF asset is missing: ${MODEL_PATH}/${required}" >&2
        exit 1
    fi
done
if [[ -n "${DATA_PATH}" ]]; then
    DATA_PATH=$(cd "$(dirname "${DATA_PATH}")" 2>/dev/null && pwd)/$(basename "${DATA_PATH}") || exit 1
    if [[ ! -s "${DATA_PATH}" || ! -s "${PROCESSOR_PATH}/inference/config.json" ]]; then
        echo "Training data or processor inference/config.json is missing." >&2
        exit 1
    fi
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -c '
import pathlib
import hyper_parallel
from hyper_parallel.models import HyperAutoModelForImageTextToText
from transformers.models.deepseek_v4 import DeepseekV4Config
expected = pathlib.Path.cwd() / "hyper_parallel"
actual = pathlib.Path(hyper_parallel.__file__).resolve().parent
if actual != expected.resolve():
    raise RuntimeError(f"Expected checkout {expected}, imported {actual}; run pip install -e .")
'
export HYPER_PARALLEL_HF_LOADER=dcp
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
if [[ -e "${OUTPUT_DIR}/baseline.started" ]]; then
    echo "Run already exists; choose a new RUN_NAME or OUTPUT_DIR: ${OUTPUT_DIR}" >&2
    exit 1
fi
ENGRAM_ASSETS_PATH="${OUTPUT_DIR}/engram_assets.json"
echo "Generating released-size Engram hashing assets from the local config and tokenizer..."
"${PYTHON_BIN}" -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_assets \
    --model-dir="${MODEL_PATH}" \
    --output="${ENGRAM_ASSETS_PATH}" \
    --released
if [[ -z "${DATA_PATH}" ]]; then
    echo "Generating the validated synthetic VLM dataset and processor assets..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_deepseek_v41_vlm_data.py" \
        --model-dir="${PROCESSOR_PATH}" \
        --output-dir="${OUTPUT_DIR}/data" \
        --num-samples="${NUM_SAMPLES:-256}"
    DATA_PATH="${OUTPUT_DIR}/data/train.jsonl"
    PROCESSOR_PATH="${OUTPUT_DIR}/data/processor_assets"
fi
echo "Loading the first two text and vision layers, including the original layer-1 Host Engram table."
touch "${OUTPUT_DIR}/baseline.started"

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${PROJECT_ROOT}/scripts/train_vl.py" \
    "${SCRIPT_DIR}/train_deepseek_v41_vlm_online_load_hf.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --model.engram_assets_path="${ENGRAM_ASSETS_PATH}" \
    --dataset.model_assets.config_path="${PROCESSOR_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --accelerator.cp_size=1 \
    --model.load_base_model=true \
    --checkpoint.restore_from=null \
    --checkpoint.save_ckpt=true \
    --checkpoint.checkpoint_dir="${OUTPUT_DIR}/checkpoints" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${RUN_NAME}.log"

touch "${OUTPUT_DIR}/baseline.success"
