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
# Multi-node generic HF text-training launcher (ssh fan-out).
#
#   NODES="host1 host2 ..." $0 /path/to/hf-model <iters> [trainer overrides...]
#
# Assumptions (documented, not hidden): the repository checkout and the model
# directory live on shared storage visible to every node at identical paths,
# and password-less ssh reaches every node from the launcher host. Rank 0
# runs on the first node in NODES and this script follows it in the
# foreground; the other nodes are started in the background.
#
# Discipline: every node must pass the admission guard before anything
# starts (all-or-nothing), and is release-verified after the run. If any
# guard reports busy, nothing is launched anywhere — back off and re-probe
# later instead of queueing.
#
# Env knobs (same as the single-node launcher, plus):
#   NODES         space-separated node hostnames/IPs (required)
#   MASTER_PORT   rendezvous port on the first node (default 29500)
#   DRYRUN=1      print per-node commands instead of executing them
#   SSH_OPTS      extra ssh options (e.g. "-o ConnectTimeout=15")

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
GUARD="${PROJECT_ROOT}/.agent/skills/hf-model-integration/scripts/node_guard.sh"
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo/hf_text_bringup"
DATA_PATH="${OUTPUT_DIR}/online_2k.jsonl"
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CANN_SETENV=${CANN_SETENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
MASTER_PORT=${MASTER_PORT:-29500}
DRYRUN=${DRYRUN:-0}
SSH_OPTS=${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=15}

if [[ -z "${NODES:-}" ]]; then
    echo "NODES is required: NODES=\"host1 host2\" $0 <model> <iters> ..." >&2
    exit 1
fi
if [[ $# -lt 2 ]]; then
    echo "Usage: NODES=\"...\" $0 /path/to/hf-model <train_iters> [overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1; ITERS=$2; shift 2
read -r -a NODE_ARR <<< "${NODES}"
NNODES=${#NODE_ARR[@]}
MASTER_ADDR=${NODE_ARR[0]}

OVERRIDES=""
for arg in "$@"; do OVERRIDES+=" $(printf '%q' "${arg}")"; done

remote_cmd() {  # $1 = node rank
    local rank=$1
    cat <<RCMD
set -e
source "${CANN_SETENV}"
export HYPER_PARALLEL_PLATFORM=torch
export HCCL_CONNECT_TIMEOUT=\${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=\${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_WHITELIST_DISABLE=\${HCCL_WHITELIST_DISABLE:-1}
export PYTORCH_NPU_ALLOC_CONF=\${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${EXTRA_PYTHONPATH:+${EXTRA_PYTHONPATH}:}${PROJECT_ROOT}"
SELF_IP=\$(hostname -I | tr " " "\n" | grep -v "^127\." | head -1)
[ -n "\${SELF_IP}" ] && export HCCL_IF_IP=\${SELF_IP}
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=${NNODES} --node-rank=${rank} \
    --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} \
    --nproc-per-node=${NPROC_PER_NODE} \
    --module examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_hf_text.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --training.train_iters="${ITERS}"${OVERRIDES}
RCMD
}

run_or_print() {  # $1 = node, rest = command string on stdin semantics
    local node=$1; shift
    if [[ "${DRYRUN}" = "1" ]]; then
        echo "--- ssh ${node} <<"; cat; echo "--- end"
    else
        # shellcheck disable=SC2086
        ssh ${SSH_OPTS} "${node}" 'bash -s'
    fi
}

echo "[dist] admission guard on ${NNODES} nodes (all-or-nothing)"
for node in "${NODE_ARR[@]}"; do
    if [[ "${DRYRUN}" = "1" ]]; then
        echo "--- guard: ssh ${node} bash ${GUARD} pre"
    else
        # shellcheck disable=SC2086
        ssh -n ${SSH_OPTS} "${node}" "bash '${GUARD}' pre" \
            || { echo "[dist] ${node} busy or undetermined - nothing launched"; exit 3; }
    fi
done

if [[ "${DRYRUN}" != "1" && ! -s "${DATA_PATH}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    "${PYTHON_BIN}" -m examples.training_demo.hf_text_bringup.prepare_text_data \
        --output "${DATA_PATH}" --num-samples 128 --sequence-length 2048
fi

mkdir -p "${OUTPUT_DIR}"
rc=0
if [[ "${DRYRUN}" = "1" ]]; then
    for i in $(seq 0 $((NNODES - 1))); do
        echo "[dist] node-rank ${i} on ${NODE_ARR[$i]}:"
        remote_cmd "${i}" | run_or_print "${NODE_ARR[$i]}"
    done
else
    for i in $(seq $((NNODES - 1)) -1 1); do
        node=${NODE_ARR[$i]}
        echo "[dist] launching node-rank ${i} on ${node}"
        remote_cmd "${i}" | run_or_print "${node}" \
            > "${OUTPUT_DIR}/dist_rank${i}_${node}.log" 2>&1 &
    done
    echo "[dist] launching node-rank 0 on ${MASTER_ADDR} (foreground)"
    remote_cmd 0 | run_or_print "${MASTER_ADDR}" || rc=$?
    wait || true
fi

echo "[dist] release verification"
for node in "${NODE_ARR[@]}"; do
    if [[ "${DRYRUN}" = "1" ]]; then
        echo "--- post: ssh ${node} bash ${GUARD} post"
    else
        # shellcheck disable=SC2086
        ssh -n ${SSH_OPTS} "${node}" "bash '${GUARD}' post" || rc=3
    fi
done
exit ${rc}
