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
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CP_SIZE=${CP_SIZE:-1}
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /local/config.json /local/indexed_prefix [--trainer.option=value ...]" >&2
    exit 2
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ && "${CP_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
        (( CP_SIZE != 1 )); then
    echo "NPROC_PER_NODE must be positive and this baseline requires CP_SIZE=1." >&2
    exit 2
fi
CONFIG_PATH=$(realpath -e "$1")
DATA_PREFIX=$(realpath -m "$2")
shift 2
if [[ ! -s "${CONFIG_PATH}" || ! -s "${DATA_PREFIX}.bin" || ! -s "${DATA_PREFIX}.idx" ]]; then
    echo "Provide an existing nonempty architecture JSON and Indexed .bin/.idx prefix." >&2
    exit 2
fi
read -r VOCAB_SIZE EOD_TOKEN_ID < <("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
text = config.get("text_config", config)
print(text["vocab_size"], text["eos_token_id"])
PY
)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
DP_SIZE=$((NPROC_PER_NODE / CP_SIZE))
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone "--nproc_per_node=${NPROC_PER_NODE}" \
    --module scripts.train_lm "${SCRIPT_DIR}/train_llm_pretrain.yaml" \
    "--model.config_path=${CONFIG_PATH}" \
    "--dataset.data_path=${DATA_PREFIX}" \
    "--dataset.model_assets.tokenizer.pretrained_model_name_or_path=$(dirname "${CONFIG_PATH}")" \
    "--dataset.model_assets.tokenizer.vocab_size=${VOCAB_SIZE}" \
    "--dataset.model_assets.tokenizer.eod_token_id=${EOD_TOKEN_ID}" \
    "--accelerator.cp_size=${CP_SIZE}" \
    "--fsdp_config.dp_shard_size=${NPROC_PER_NODE}" \
    "--fsdp_config.edp_shard_size=${NPROC_PER_NODE}" \
    "--training.global_batch_size=${DP_SIZE}" "$@"
