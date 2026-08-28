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
VARIANT=${1:-ulysses}
if [[ $# -gt 0 ]]; then
    shift
fi

case "${VARIANT}" in
    ulysses)
        CONFIG=examples/qwen3_moe_tp_ep_cp/train_tp2_ep2_cp2_ulysses.yaml
        RDZV_ID=qwen3_tp_ep_cp_ulysses
        MASTER_PORT=${MASTER_PORT:-29531}
        ;;
    hybrid)
        CONFIG=examples/qwen3_moe_tp_ep_cp/train_tp2_ep2_cp4_hybrid.yaml
        RDZV_ID=qwen3_tp_ep_cp_hybrid
        MASTER_PORT=${MASTER_PORT:-29532}
        ;;
    *)
        echo "Usage: $0 [ulysses|hybrid] [training overrides...]" >&2
        exit 1
        ;;
esac

export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

torchrun \
    --nproc_per_node=8 \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT}" \
    --module examples.training_demo.train_text \
    "${CONFIG}" \
    "$@"
