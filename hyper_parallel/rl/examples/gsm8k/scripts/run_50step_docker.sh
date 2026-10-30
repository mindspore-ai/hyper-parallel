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

framework=${1:-hyper}
case "$framework" in hyper|codex) ;; *) echo 'Usage: run_50step_docker.sh [hyper|codex]' >&2; exit 2 ;; esac
if (( $# > 1 )); then echo 'Use the saved YAML for explicit configuration changes.' >&2; exit 2; fi
: "${RL_MODEL:?Set RL_MODEL to the Qwen3-4B directory}"
: "${RL_DATA:?Set RL_DATA to the prepared 6400/500-question parquet directory}"
: "${RL_RESULTS:?Set RL_RESULTS to a new result directory}"
repo=$(git rev-parse --show-toplevel)
model=$(realpath "$RL_MODEL")
data=$(realpath "$RL_DATA")
results=$(realpath -m "$RL_RESULTS")
[[ -f "$model/config.json" && -f "$data/train.parquet" && -f "$data/test.parquet" ]]
# Never mix an earlier completed or interrupted run into a new experiment.
mkdir "$results"
image=${RL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64}
docker image inspect "$image" --format '{{.Id}}' > "$results/image_id.txt"
git -C "$repo" rev-parse HEAD > "$results/source_commit.txt"
git -C "$repo" diff HEAD --binary > "$results/source.patch"
# Include unstaged and untracked source, excluding ignored logs, models and checkpoints.
(
    cd "$repo"
    git ls-files --cached --others --exclude-standard -z |
        while IFS= read -r -d '' file; do
            if [[ -f "$file" || -L "$file" ]]; then printf '%s\0' "$file"; fi
        done | tar --null -T - -cf "$results/source.tar"
)
sha256sum "$results/source.tar" > "$results/source_sha256.txt"
mkdir "$results/source"
tar -xf "$results/source.tar" -C "$results/source"
tools="$results/source/hyper_parallel/rl/examples/gsm8k/scripts"
export RL_RUN_RESULTS="$results"
PYTHONPATH="$tools${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import time
from monitor_8npu_run import idle, sample

root = Path(os.environ['RL_RUN_RESULTS'])
deadline = time.monotonic() + int(os.environ.get('RL_WAIT_TIMEOUT', '3600'))
consecutive = 0
while consecutive < 3:
    if (root / 'PAUSE').exists() or time.monotonic() >= deadline:
        raise RuntimeError('Paused or timed out waiting for eight idle devices')
    row = sample()
    with (root / 'idle-monitor.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    available = idle(row)
    consecutive = consecutive + 1 if available else 0
    time.sleep(3)
for device in range(8):
    state = subprocess.run(['npu-smi', 'info', '-t', 'device-share', '-i', str(device), '-c', '0'],
                           check=True, capture_output=True, text=True).stdout
    if 'False' not in state:
        raise RuntimeError('This recipe requires device_share=false on every device')
PY
identity=$(docker create --privileged --network host --shm-size 64g \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$results/source:/repo" -v "$results:/results" \
    -v "$model:/models/Qwen3-4B:ro" -v "$data:/data:ro" -w /repo \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e GLOO_SOCKET_IFNAME=lo \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
    -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= -e http_proxy= -e https_proxy= -e all_proxy= \
    "$image" bash /repo/hyper_parallel/rl/examples/gsm8k/scripts/run_50step_inside.sh "$framework")
printf '%s\n' "$identity" > "$results/container_id.txt"
finish() {
    touch "$results/stop_monitor"
    docker stop -t 10 "$identity" >/dev/null
}
trap finish EXIT
python3 "$tools/monitor_8npu_run.py" --container-id "$identity" --results "$results" --case-dir "$results" \
    > "$results/monitor.log" 2>&1 &
monitor_pid=$!
docker start -a "$identity" > "$results/launcher.log" 2>&1 &
runner_pid=$!
while kill -0 "$runner_pid" 2>/dev/null; do
    if ! kill -0 "$monitor_pid" 2>/dev/null && [[ ! -e "$results/stop_monitor" ]]; then
        echo 'Device monitor exited unexpectedly; stopping this experiment.' >&2
        exit 1
    fi
    sleep 2
done
status=0
wait "$runner_pid" || status=$?
touch "$results/stop_monitor"
wait "$monitor_pid" || status=1
if [[ -f "$results/interruption.json" ]]; then status=1; fi
printf '%s\n' "$status" > "$results/launcher_exit_code"
echo "Run ended with exit code $status. Results: $results"
exit "$status"
