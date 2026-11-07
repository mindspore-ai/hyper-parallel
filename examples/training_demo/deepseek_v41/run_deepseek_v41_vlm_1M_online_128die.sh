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
set -eo pipefail

PREPARE_ONLY=0
if [[ ${1:-} == --prepare-resources ]]; then
    PREPARE_ONLY=1
    shift
fi
TIME=${TIME:-$(date +"%Y%m%d_%H%M%S")}
MODEL_PATH=${MODEL_PATH:-/path/to/DeepSeek-V4.1-Flash}
DATA_PATH=${DATA_PATH:-/path/to/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-output/training_demo/deepseek_v41_logs_${TIME}}
PARAMETER_DIVISOR=${PARAMETER_DIVISOR:-4}
ASSETS_PATH=${ASSETS_PATH:-output/training_demo/deepseek_v41/artifacts/engram_depth_preserving_d${PARAMETER_DIVISOR}.json}
CONFIG_PATH=${CONFIG_PATH:-examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_1M online_128die.yaml}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python}
NNODES=${NNODES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
NODE_RANK=${NODE_RANK:?Use the cluster controller or set NODE_RANK}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_ADDR=${NODE_ADDR:?Set NODE_ADDR to the reachable hostfile IP of this node}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
mkdir -p "${OUTPUT_DIR}"

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/cann-9.2.0/opp/vendors/omni_training_custom_transformer/bin/set_env.bash

export ASCEND_CUSTOM_OPP_PATH="/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="/workspace/venv/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash

set -u

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

TRAINER_SITE_PACKAGES=$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
case "${PYTHONPATH:-}" in
    "${TRAINER_SITE_PACKAGES}" | "${TRAINER_SITE_PACKAGES}":*) ;;
    *) export PYTHONPATH="${TRAINER_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

print(f"Training Python: {sys.executable}", flush=True)
spec = importlib.util.find_spec("cann_ops_transformer")
if spec is None:
    raise ModuleNotFoundError("Install the matching cann_ops_transformer wheel into this Python environment")
print(f"cann_ops_transformer: {spec.origin}", flush=True)
PY

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

if [[ -z "${TORCH_EXTENSIONS_DIR:-}" ]]; then
    TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/hyper_parallel_csa2.XXXXXXXX)
fi
export TORCH_EXTENSIONS_DIR
printf 'Torch extensions cache: %s\n' "${TORCH_EXTENSIONS_DIR}"

if [[ ! -s "${ASSETS_PATH}" ]]; then
    "${PYTHON_BIN}" -m hyper_parallel.models.deepseek_v41.tools.prepare_deepseek_v41_assets \
        --model-dir "${MODEL_PATH}" \
        --output "${ASSETS_PATH}" \
        --bucket-base 4096 \
        --parameter-divisor "${PARAMETER_DIVISOR}" \
        --table-pad-multiple 128
fi
if [[ ! -s "${DATA_PATH}" ]]; then
    if [[ "${DATA_PATH##*/}" != train.jsonl ]]; then
        echo "DATA_PATH does not exist: ${DATA_PATH}" >&2
        echo "Use an existing VLM dataset or a path ending in /train.jsonl." >&2
        exit 1
    fi
    "${PYTHON_BIN}" -m hyper_parallel.models.deepseek_v41.tools.prepare_deepseek_v41_vlm_data \
        --model-dir "${MODEL_PATH}" \
        --output-dir "$(dirname "${DATA_PATH}")" \
        --num-samples 128000 \
        --sample-format image_caption
fi

if ((PREPARE_ONLY)); then
    exit 0
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    --local-addr="${NODE_ADDR}" --max_restarts=0 \
    --module scripts.train_vl \
    "${CONFIG_PATH}" \
    --model.config_path="${MODEL_PATH}" \
    --dataset.model_assets.config_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --profiler.output_path="${OUTPUT_DIR}/profiler" \
    --memory.save_path="${OUTPUT_DIR}/memory_profiler" \
    --model.engram_assets_path="${ASSETS_PATH}" \
    --training.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    --model.text_parameter_divisor="${PARAMETER_DIVISOR}" \
    --model.vision_parameter_divisor="${PARAMETER_DIVISOR}" \
    "$@"
