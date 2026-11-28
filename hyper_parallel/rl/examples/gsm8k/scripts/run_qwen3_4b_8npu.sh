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

# Run the full eight-device Qwen3-4B GSM8K recipe from the mounted checkout.
set -euo pipefail
mode=${1:-}
registry=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl
mounts=()
case "$mode" in
    native) image="$registry:v0.23.0.post1-code-agent-arm64" ;;
    unified)
        image="$registry:consistency-v0.30.0-cann9.2.1-arm64"
        wheel=$(realpath -- "${V4_WHEEL:?Set V4_WHEEL to the patched V4 attention wheel}")
        [[ -f "$wheel" ]]
        mounts+=(-v "$wheel:/wheels/$(basename -- "$wheel"):ro" -e HCCL_DETERMINISTIC=true)
        ;;
    *) echo "Usage: $0 {native|unified} [--section.field=value ...]" >&2; exit 2 ;;
esac
shift
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$scripts/../../../../.." && pwd)
model=$(realpath -- "${GSM8K_MODEL:?Set GSM8K_MODEL to the full Qwen3-4B model directory}")
data=$(realpath -- "${GSM8K_DATA:?Set GSM8K_DATA to the prepared 6400/500 splits}")
[[ -f "$model/config.json" ]]
for split in train test; do
    [[ -f "$data/$split.parquet" ]] || { echo "Missing $data/$split.parquet" >&2; exit 1; }
done
if [[ -n "${EXTRA_WHEELS:-}" ]]; then
    mounts+=(-v "$(realpath -- "$EXTRA_WHEELS"):/deps:ro" -e PIP_NO_INDEX=1 -e PIP_FIND_LINKS=/deps)
fi
result=$(realpath -m -- "${GSM8K_RESULTS:?Set GSM8K_RESULTS to a new run directory}")
mkdir -p -- "$(dirname -- "$result")"
mkdir -- "$result"
docker run --rm --privileged --network host --shm-size 64g \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$repo:/repo:ro" -v "$model:/models/Qwen3-4B:ro" -v "$data:/data:ro" -v "$result:/results" \
    "${mounts[@]}" -w /repo -e MODE="$mode" -e RESULT_OWNER="$(id -u):$(id -g)" \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e OMP_NUM_THREADS=4 -e TASK_QUEUE_ENABLE=2 -e TOKENIZERS_PARALLELISM=false \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e GLOO_SOCKET_IFNAME=lo \
    -e VLLM_HOST_IP=127.0.0.1 -e VLLM_ASCEND_ENABLE_NZ=0 -e VLLM_SERVER_DEV_MODE=1 -e VLLM_BATCH_INVARIANT=0 \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e HYPER_RL_FAILURE_DIR=/results \
    -e HCCL_IF_BASE_PORT=64600 -e HCCL_NPU_SOCKET_PORT_RANGE=64600-64799 \
    "$image" bash -c '
        set -eo pipefail
        trap "chown -R $RESULT_OWNER /results" EXIT
        export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}
        unset PYTORCH_NPU_ALLOC_CONF
        if [[ "$MODE" == unified ]]; then
            source /usr/local/Ascend/cann/set_env.sh
            sha256sum /wheels/*.whl
            python -m pip install --no-index --no-deps --target /opt/hp-v4 /wheels/*.whl
            export PYTHONPATH=/opt/hp-v4:$PYTHONPATH
        else
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            python hyper_parallel/rl/docker/patches/patch_sampler.py --record /tmp/sampler.json
            fix=/repo/hyper_parallel/rl/docker/patches/vllm-dp-coordinator-timeout.patch
            if patch --batch --forward --dry-run -p1 -d /vllm-workspace/vllm < "$fix" >/dev/null; then
                patch --batch --forward -p1 -d /vllm-workspace/vllm < "$fix"
            else
                patch --force --dry-run -R -p1 -d /vllm-workspace/vllm < "$fix"
            fi
        fi
        mkdir /tmp/editable
        cp setup.py requirements.txt README.md MANIFEST.in /tmp/editable/
        ln -s /repo/hyper_parallel /tmp/editable/hyper_parallel
        python -m pip install --no-deps --no-build-isolation -e /tmp/editable
        python -m pip install --no-deps peft==0.21.2 accelerate==1.12.0
        bash hyper_parallel/rl/docker/install_runtime.sh
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
            --master_addr=127.0.0.1 --master_port=64599 hyper_parallel/rl/train_rl.py "hyper_parallel/rl/examples/gsm8k/configs/qwen3_4b_gsm8k_${MODE}_50step.yaml" "$@"
    ' gsm8k "$@" 2>&1 | tee "$result/train.log"
