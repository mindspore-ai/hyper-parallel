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
framework=${1:?Expected hyper or codex}
trap 'touch /results/stop_monitor' EXIT
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# The trainer uses queue mode 2; Hyper vLLM children use the YAML override for graph capture.
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=2
export VLLM_HOST_IP=127.0.0.1 VLLM_ASCEND_ENABLE_NZ=0 VLLM_SERVER_DEV_MODE=1 VLLM_BATCH_INVARIANT=0
unset PYTORCH_NPU_ALLOC_CONF
base=/repo/hyper_parallel/rl/examples/gsm8k
if [[ "$framework" == hyper || "$framework" == codex ]]; then
    export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}
    export HCCL_IF_BASE_PORT=64600 HCCL_NPU_SOCKET_PORT_RANGE=64600-64799
    export HYPER_RL_FAILURE_DIR=/results
    cd /repo
    git config --global --add safe.directory /repo
    python -m pip install --no-deps --no-build-isolation -e .
    bash hyper_parallel/rl/docker/install_runtime.sh
    python "/repo/hyper_parallel/rl/docker/patches/patch_sampler.py" --record /results/sampler.json
    startup_patch=/repo/hyper_parallel/rl/docker/patches/vllm-dp-coordinator-timeout.patch
    if patch --dry-run -d /vllm-workspace/vllm -p1 < "$startup_patch" >/dev/null; then
        patch -d /vllm-workspace/vllm -p1 < "$startup_patch"
    elif ! patch --dry-run -R -d /vllm-workspace/vllm -p1 < "$startup_patch" >/dev/null; then
        echo 'Incompatible vLLM startup patch' >&2
        exit 1
    fi
    python - <<'PYCODE'
import hyper_parallel
if not hyper_parallel.__file__.startswith('/repo/'):
    raise RuntimeError('Editable installation does not point at the source snapshot')
print(hyper_parallel.__file__)
PYCODE
    recipe=qwen3_4b_gsm8k_50step.yaml
    if [[ "$framework" == codex ]]; then
        recipe=qwen3_4b_gsm8k_codex_50step.yaml
        codex --version > /results/codex_version.txt
    fi
    cp "$base/configs/$recipe" /results/config.yaml
    command=(python -m torch.distributed.run --nnodes=1 --nproc_per_node=8
        --master_addr=127.0.0.1 --master_port=64599 /repo/hyper_parallel/rl/train_rl.py /results/config.yaml)
else
    echo 'Expected hyper or codex' >&2
    exit 2
fi
set -u
sha256sum /data/train.parquet /data/test.parquet > /results/data_sha256.txt
python -m pip freeze > /results/packages.txt
set +e
date +%s > /results/process_started
"${command[@]}" > /results/train.log 2>&1
status=$?
date +%s > /results/process_finished
printf '%s\n' "$status" > /results/exit_code
exit "$status"
