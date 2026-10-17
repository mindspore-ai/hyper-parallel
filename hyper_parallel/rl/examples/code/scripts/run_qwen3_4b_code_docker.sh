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

# Keep engine environment variables inside the container: host cleanup scripts
# may match their names in the Docker client's command line.
if [[ "${1:-}" == --container ]]; then
    shift
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
    trap 'chown -R "${HYPER_CODE_UID}:${HYPER_CODE_GID}" /results' EXIT
    cd /repo
    git config --global --add safe.directory /repo
    python -m pip install --no-deps --no-build-isolation -e .
    bash hyper_parallel/rl/docker/install_runtime.sh
    export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_HOST_IP=127.0.0.1
    export GLOO_SOCKET_IFNAME=lo
    unset VLLM_PLUGINS
    python - <<'PYTHON'
import hyper_parallel
import torch
import torch_npu

print(f"HyperParallel source: {hyper_parallel.__file__}")
if torch.npu.device_count() != 4:
    raise RuntimeError("Expected four visible NPUs; check driver allocation and device sharing")
PYTHON
    python -m torch.distributed.run --nnodes=1 --node_rank=0 --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="${HYPER_CODE_MASTER_PORT}" \
        hyper_parallel/rl/train_rl.py \
        hyper_parallel/rl/examples/code/configs/qwen3_4b_code_vllm.yaml \
        --train.checkpoint.output_dir=/results/checkpoints \
        --logging.wandb.directory=/results/wandb \
        "--agentic.code.endpoint=${HYPER_CODE_ENDPOINT}" \
        "--agentic.code.runtime_version=${HYPER_CODE_SANDBOX_ID}" \
        "$@" 2>&1 | tee /results/train.log
    exit 0
fi

set -u
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: run_qwen3_4b_code_docker.sh [--dry-run] [--config.key=value ...]

Run the four-NPU Qwen3-4B code recipe against an already deployed local sandbox.
Required: HYPER_CODE_DATA_ROOT (reviewed train.parquet and validation.parquet).
Optional environment:
  HYPER_CODE_MODEL_ROOT       Default: ../models/Qwen3-4B beside the repository
  HYPER_CODE_OUTPUT_DIR       Default: output/code/<timestamp>-<pid> in the repository
  HYPER_CODE_DEVICES          Default: 4,5,6,7 (four ascending physical NPU IDs)
  HYPER_CODE_PRIVILEGED       Default: 0; set 1 explicitly if the driver requires it
  HYPER_CODE_IMAGE            Default: unified v0.23.0.post1-code-agent-arm64
  HYPER_CODE_SANDBOX_IMAGE    Default: SWR huawei-hyper-rl/sandboxfusion-python:v2-arm64
  HYPER_CODE_SANDBOX_CONTAINER Default: hyper-rl-code-sandbox
  HYPER_CODE_ENDPOINT         Default: http://127.0.0.1:18790
  HYPER_CODE_MASTER_PORT      Default: 62399
  HYPER_CODE_HCCL_BASE_PORT   Default: 62400 (uses this port through +99)

Extra arguments override existing YAML keys without changing recipe defaults.
--dry-run checks prerequisites and prints the command without starting training.
This script does not download assets, change device sharing, or exempt host cleanup.
HELP
    exit 0
fi
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=1
    shift
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../../.." && pwd)
: "${HYPER_CODE_DATA_ROOT:?Set HYPER_CODE_DATA_ROOT to reviewed prepared data}"
: "${HYPER_CODE_MODEL_ROOT:=${repo_root}/../models/Qwen3-4B}"
: "${HYPER_CODE_OUTPUT_DIR:=${repo_root}/output/code/$(date +%Y%m%d-%H%M%S)-$$}"
: "${HYPER_CODE_DEVICES:=4,5,6,7}"
: "${HYPER_CODE_PRIVILEGED:=0}"
: "${HYPER_CODE_IMAGE:=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64}"
: "${HYPER_CODE_SANDBOX_IMAGE:=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v2-arm64}"
: "${HYPER_CODE_SANDBOX_CONTAINER:=hyper-rl-code-sandbox}"
: "${HYPER_CODE_ENDPOINT:=http://127.0.0.1:18790}"
: "${HYPER_CODE_MASTER_PORT:=62399}"
: "${HYPER_CODE_HCCL_BASE_PORT:=62400}"

fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ "${HYPER_CODE_DEVICES}" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*),(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]] || fail 'Select exactly four NPU IDs'
IFS=',' read -r -a devices <<< "${HYPER_CODE_DEVICES}"
previous=-1
device_args=()
for device in "${devices[@]}"; do
    (( 10#${device} > previous )) || fail 'NPU IDs must be distinct and ascending'
    previous=$((10#${device}))
    device_args+=(--device "/dev/davinci${device}")
done
[[ "${HYPER_CODE_PRIVILEGED}" =~ ^[01]$ ]] || fail 'HYPER_CODE_PRIVILEGED must be 0 or 1'
for port in "${HYPER_CODE_MASTER_PORT}" "${HYPER_CODE_HCCL_BASE_PORT}"; do
    [[ "${port}" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || fail 'Ports must be integers in 1..65535'
done
(( HYPER_CODE_HCCL_BASE_PORT <= 65436 )) || fail 'HCCL port range must end at or below 65535'
(( HYPER_CODE_MASTER_PORT < HYPER_CODE_HCCL_BASE_PORT ||
   HYPER_CODE_MASTER_PORT > HYPER_CODE_HCCL_BASE_PORT + 99 )) || fail 'Master and HCCL ports must not overlap'
visible_devices=0,1,2,3
if [[ "${HYPER_CODE_PRIVILEGED}" == 1 ]]; then
    device_args=(--privileged)
    visible_devices=${HYPER_CODE_DEVICES}
else
    device_args+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
fi
[[ -f "${HYPER_CODE_MODEL_ROOT}/config.json" ]] || fail 'Model config.json not found'
for split in train validation; do
    [[ -f "${HYPER_CODE_DATA_ROOT}/${split}.parquet" ]] || fail "Reviewed ${split}.parquet not found"
done
model_root=$(cd -- "${HYPER_CODE_MODEL_ROOT}" && pwd)
data_root=$(cd -- "${HYPER_CODE_DATA_ROOT}" && pwd)
docker image inspect "${HYPER_CODE_IMAGE}" >/dev/null
sandbox_id=$(docker image inspect --format '{{.Id}}' "${HYPER_CODE_SANDBOX_IMAGE}")
running_sandbox=$(docker inspect --format '{{.State.Running}} {{.Image}}' "${HYPER_CODE_SANDBOX_CONTAINER}")
[[ "${running_sandbox}" == "true ${sandbox_id}" ]] || fail 'Sandbox container must run the selected image'
sandbox_address=$(docker port "${HYPER_CODE_SANDBOX_CONTAINER}" 8080/tcp)
[[ "${HYPER_CODE_ENDPOINT%/}" == "http://${sandbox_address}" && "${sandbox_address}" == 127.0.0.1:* ]] ||
    fail 'Endpoint must match the inspected sandbox container loopback port'
curl --fail --silent --show-error --max-time 10 "${HYPER_CODE_ENDPOINT%/}/health" >/dev/null
output_dir=$(realpath -m -- "${HYPER_CODE_OUTPUT_DIR}")
command=(docker run --rm --runtime runc --network host --shm-size 64g "${device_args[@]}"
    -e "ASCEND_RT_VISIBLE_DEVICES=${visible_devices}"
    -e "HYPER_CODE_UID=$(id -u)" -e "HYPER_CODE_GID=$(id -g)"
    -e "HYPER_CODE_MASTER_PORT=${HYPER_CODE_MASTER_PORT}"
    -e "HCCL_IF_BASE_PORT=${HYPER_CODE_HCCL_BASE_PORT}"
    -e "HCCL_NPU_SOCKET_PORT_RANGE=${HYPER_CODE_HCCL_BASE_PORT}-$((HYPER_CODE_HCCL_BASE_PORT + 99))"
    -e "HYPER_CODE_ENDPOINT=${HYPER_CODE_ENDPOINT}" -e "HYPER_CODE_SANDBOX_ID=${sandbox_id}"
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro
    -v "${repo_root}:/repo" -v "${model_root}:/models/Qwen3-4B:ro"
    -v "${data_root}:/data/code-stdio:ro" -v "${output_dir}:/results"
    -w /repo --entrypoint /bin/bash "${HYPER_CODE_IMAGE}"
    /repo/hyper_parallel/rl/examples/code/scripts/run_qwen3_4b_code_docker.sh --container "$@")
if (( dry_run )); then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    mkdir -p -- "${output_dir}"
    printf 'Training log: %s/train.log\n' "${output_dir}"
    exec "${command[@]}"
fi
