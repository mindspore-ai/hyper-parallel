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
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 /local/config.json /local/processor /local/train.jsonl [--trainer.option=value ...]" >&2
    exit 2
fi
if [[ "${NPROC_PER_NODE}" != 1 && "${NPROC_PER_NODE}" != 2 ]]; then
    echo "This K3 launcher supports only NPROC_PER_NODE=1 or 2." >&2
    exit 2
fi
CONFIG_PATH=$1
PROCESSOR_PATH=$2
DATA_PATH=$3
shift 3

CONFIG_PATH=$(realpath -e "${CONFIG_PATH}")
PROCESSOR_PATH=$(realpath -e "${PROCESSOR_PATH}")
DATA_PATH=$(realpath -e "${DATA_PATH}")
if [[ ! -s "${CONFIG_PATH}" || ! -d "${PROCESSOR_PATH}" || ! -s "${DATA_PATH}" ]]; then
    echo "Provide an existing nonempty architecture JSON, processor directory, and JSONL dataset." >&2
    exit 2
fi
if [[ ! -s "${PROCESSOR_PATH}/tokenizer_config.json" ]]; then
    echo "Local tokenizer_config.json is missing from ${PROCESSOR_PATH}" >&2
    exit 2
fi

# Local processor Python modules must have been reviewed and installed with its dependencies.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
FSDP_OVERRIDES=()
if [[ "${NPROC_PER_NODE}" == 2 ]]; then
    FSDP_OVERRIDES=(--fsdp_config.dp_shard_size=2 --fsdp_config.edp_shard_size=2 --training.global_batch_size=2)
fi
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone "--nproc_per_node=${NPROC_PER_NODE}" \
    --module scripts.train_vl "${SCRIPT_DIR}/train_image_pretrain.yaml" \
    "--model.config_path=${CONFIG_PATH}" \
    "--dataset.model_assets.pretrained_model_name_or_path=${PROCESSOR_PATH}" \
    "--dataset.data_path=${DATA_PATH}" \
    "${FSDP_OVERRIDES[@]}" \
    "$@"
