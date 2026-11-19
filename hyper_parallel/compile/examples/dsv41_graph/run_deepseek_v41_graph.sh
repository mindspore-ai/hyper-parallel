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

# Example: MODEL_DIR=/models/v41 TOKENIZER_DIR=/models/tokenizer \
# WIKITEXT_TRAIN_FILE=/datasets/wikitext-train.jsonl bash "$0"
# Model config must contain compatible V4.1/Engram fields with active Compressor
# and Indexer roles (compression ratios, KV/Index source layers).
# Defaults to eight ranks with FSDP4 + TP2/SP + EP2 (expert EDP4).
# TP_SIZE=1 disables SP; EP_SIZE=1 disables expert parallelism.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
: "${MODEL_DIR:?Set MODEL_DIR to a compatible local model configuration directory}"
: "${TOKENIZER_DIR:?Set TOKENIZER_DIR to a local tokenizer directory}"
: "${WIKITEXT_TRAIN_FILE:?Set WIKITEXT_TRAIN_FILE to a local WikiText JSONL/Arrow/Parquet file}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/train_deepseek_v41_graph_fsdp_tp_ep.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ||
      ! "${TP_SIZE}" =~ ^[1-9][0-9]*$ || ! "${EP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE, TP_SIZE and EP_SIZE must be positive integers" >&2
    exit 1
fi
if (( NPROC_PER_NODE % TP_SIZE || NPROC_PER_NODE % EP_SIZE )); then
    echo "NPROC_PER_NODE must divide evenly by TP_SIZE and EP_SIZE" >&2
    exit 1
fi
DP_SIZE=$((NPROC_PER_NODE / TP_SIZE))
EDP_SIZE=$((NPROC_PER_NODE / EP_SIZE))
SEQUENCE_PARALLEL=false
if (( TP_SIZE > 1 )); then
    SEQUENCE_PARALLEL=true
fi
OUTPUT_LOG="${OUTPUT_LOG:-output/dsv41_graph/train.log}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-600}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-HOST}"
export HCCL_DETERMINISTIC="${HCCL_DETERMINISTIC:-true}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

for file in "${MODEL_DIR}/config.json" "${WIKITEXT_TRAIN_FILE}"; do
    if [[ ! -s "${file}" ]]; then
        echo "Missing input file: ${file}" >&2
        exit 1
    fi
done
if [[ ! -d "${TOKENIZER_DIR}" ]]; then
    echo "Missing tokenizer directory: ${TOKENIZER_DIR}" >&2
    exit 1
fi
# Without this override, the model reads v41_engram_assets_path from its config.
engram_args=()
if [[ -n "${ENGRAM_ASSETS_FILE:-}" ]]; then
    if [[ ! -s "${ENGRAM_ASSETS_FILE}" ]]; then
        echo "Missing Engram assets: ${ENGRAM_ASSETS_FILE}" >&2
        exit 1
    fi
    overrides="$(python - "${CONFIG}" "${ENGRAM_ASSETS_FILE}" <<'PY_OVERRIDES'
import json
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    overrides = dict(yaml.safe_load(stream).get("model", {}).get("config_overrides", {}))
overrides["v41_engram_assets_path"] = sys.argv[2]
print(json.dumps(overrides))
PY_OVERRIDES
)"
    engram_args+=("--model.config_overrides=${overrides}")
fi
mkdir -p "$(dirname "${OUTPUT_LOG}")"
python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT:-29743}" \
    scripts/train_lm.py "${CONFIG}" \
    --model.pretrained_model_name_or_path="${MODEL_DIR}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${TOKENIZER_DIR}" \
    --dataset.data_path="${WIKITEXT_TRAIN_FILE}" \
    --accelerator.tp_size="${TP_SIZE}" \
    --accelerator.sequence_parallel="${SEQUENCE_PARALLEL}" \
    --accelerator.ep_size="${EP_SIZE}" \
    --fsdp_config.dp_shard_size="${DP_SIZE}" \
    --fsdp_config.edp_shard_size="${EDP_SIZE}" \
    --training.global_batch_size="${NPROC_PER_NODE}" \
    "${engram_args[@]}" "$@" 2>&1 | tee "${OUTPUT_LOG}"
