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

# Run inside the prepared Ascend environment. Each invocation owns a new result directory.
set -euo pipefail
mode=${1:-}
if (( $# )); then shift; fi
steps=3
if (( $# )) && [[ $1 != --* ]]; then
    steps=$1
    shift
fi
if [[ ! $mode =~ ^(native|unified)$ || ! $steps =~ ^(2|3|50)$ ]]; then
    echo "Usage: $0 {native|unified} [2|3|50] [--section.field=value ...]" >&2
    exit 2
fi

scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$scripts/../../../../.." && pwd)
config="$scripts/../configs/qwen3_30b_a3b_gsm8k_${mode}_16npu.yaml"
model=$(realpath -- "${GSM8K_MODEL:-/models/Qwen3-30B-A3B}")
data=$(realpath -- "${GSM8K_DATA:-/data/gsm8k}")
result=$(realpath -m -- "${GSM8K_RESULTS:-$PWD/results}/${mode}-${steps}step-$(date +%Y%m%d-%H%M%S)")
for file in "$model/config.json" "$data/train.parquet" "$data/test.parquet"; do
    [[ -r "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
command -v npu-smi >/dev/null
mkdir -p -- "$(dirname -- "$result")"
mkdir -- "$result"
exec > >(tee "$result/train.log") 2>&1
echo "Run directory: $result"
cd "$repo"

export PYTHONPATH="$repo/hyper_parallel/rl:$repo:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo VLLM_SERVER_DEV_MODE=1
export VLLM_ASCEND_ENABLE_NZ=0 VLLM_BATCH_INVARIANT=0
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64600}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-64600-64799}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export HYPER_RL_VLLM_CPU_THREADS=${HYPER_RL_VLLM_CPU_THREADS:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export HYPER_RL_FAILURE_DIR="$result"
if [[ "$mode" == unified ]]; then export HCCL_DETERMINISTIC=true; fi

# Use wall-clock timestamps to align device samples with evaluation, rollout and optimizer phases.
sample_resources() {
    while true; do
        date --iso-8601=seconds
        awk '/^(MemTotal|MemAvailable|SwapFree):/' /proc/meminfo
        head -n 1 /proc/stat
        for path in /sys/fs/cgroup/memory.{current,peak,max}; do
            if [[ -r "$path" ]]; then printf '%s %s\n' "$path" "$(< "$path")"; fi
        done
        npu-smi info || echo "npu-smi sample failed"
        sleep 10
    done
}
finish() {
    local status=$?
    if [[ -n ${monitor_pid:-} ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" > "$result/exit_code"
    date --iso-8601=seconds >> "$result/run-info.txt"
}
trap finish EXIT
{
    date --iso-8601=seconds
    git rev-parse HEAD
    printf 'mode=%s steps=%s devices=%s model=%s data=%s\n' "$mode" "$steps" "$ASCEND_RT_VISIBLE_DEVICES" "$model" "$data"
    sha256sum "$config" "$model/config.json" "$data/train.parquet" "$data/test.parquet"
    df -h "$result"
    python - "$model/config.json" <<'PY'
import json
import sys
from importlib.metadata import version
import torch
import torch_npu

with open(sys.argv[1]) as f:
    model = json.load(f)
print("model:", model["model_type"], "layers:", model["num_hidden_layers"], "experts:", model.get("num_experts"))
if (model["model_type"], model["num_hidden_layers"], model.get("num_experts")) != ("qwen3_moe", 48, 128):
    raise SystemExit("Expected the full Qwen3-30B-A3B checkpoint")
for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
    print(name, version(name))
count = torch.npu.device_count()
print("Visible NPUs:", count)
if count != 16:
    raise SystemExit(f"Expected 16 visible NPUs, got {count}")
PY
} > "$result/run-info.txt"
save_final=false
if [[ "$steps" == 50 ]]; then save_final=true; fi
# Paths and checkpoint cadence follow the run directory and selected step count.
overrides=(
    "$@"
    "--model.weights_path=$model" "--model.tokenizer_path=$model"
    "--data.train_path=$data/train.parquet" "--data.test_path=$data/test.parquet"
    "--train.max_steps=$steps" "--train.checkpoint.save_final=$save_final"
    "--train.checkpoint.output_dir=$result/checkpoints"
)
printf 'override=%s\n' "${overrides[@]}" >> "$result/run-info.txt"
threads=$(python -m rl.utils.cpu_threads --config "$config" --local-world-size 16 --report "$result/cpu-threads.json" "${overrides[@]}")
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
sample_resources > "$result/resources.log" 2>&1 &
monitor_pid=$!
python -m torch.distributed.run --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc-per-node=16 \
    hyper_parallel/rl/train_rl.py "$config" "${overrides[@]}"
