#!/usr/bin/env bash
set -euo pipefail
rl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
repo_root="$(cd "${rl_root}/../.." && pwd)"
export PYTHONPATH="${rl_root}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
config="${1:-${rl_root}/examples/async_training/configs/qwen3_4b_gsm8k_grpo.yaml}"
if (( $# > 0 )); then shift; fi
# The supplied recipe uses two training and two disjoint rollout devices.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
exec python -m torch.distributed.run --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29500}" \
    --nproc-per-node="${TRAIN_PROCESSES:-2}" \
    "${rl_root}/examples/async_training/launch.py" "${config}" "$@"
