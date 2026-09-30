#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}/../../.."
devices=${1:-2,3}
steps=${2:-50}
data_dir=${3:-}
if [[ -z "$data_dir" ]]; then
    echo "Usage: $0 DEVICE_IDS STEPS DATA_DIRECTORY" >&2
    exit 2
fi
if [[ ! "$steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "Steps must be a positive integer, got: $steps" >&2
    exit 2
fi
output="output/search_r1/search_r1_${steps}step_$(date +%Y%m%d_%H%M%S)_$$"
npu-smi info
read -r -p "Confirm NPUs ${devices} are reserved for ${steps} training steps; type yes: " confirmed
if [[ "$confirmed" != yes ]]; then
    echo "Cancelled; no training started."
    exit 1
fi
echo "Training output: ${PWD}/${output}"
python3 examples/search_r1/launcher.py --devices "$devices" --steps "$steps" \
    --backend-privileged --data "$data_dir" --output "$output"
python3 examples/search_r1/plot_reward.py "$output/train.log"
echo "Reward plots: ${PWD}/${output}/reward_max.svg and reward_mean.svg"
