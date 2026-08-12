#!/usr/bin/env bash

set -euo pipefail

hyper_rl_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
hyper_rl_example_root="$(cd -- "${hyper_rl_script_dir}/.." && pwd)"
hyper_rl_project_root="$(cd -- "${hyper_rl_example_root}/.." && pwd)"
hyper_rl_repo_root="$(cd -- "${hyper_rl_project_root}/../.." && pwd)"

hyper_rl_cann_env="${CANN_ENV_SCRIPT:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
hyper_rl_torchrun="${TORCHRUN_BIN:-torchrun}"
hyper_rl_config="${HYPER_RL_CONFIG:-${hyper_rl_example_root}/configs/qwen3_5_0_8b_gsm8k.yaml}"

hyper_rl_nnodes="${NNODES:-1}"
hyper_rl_node_rank="${NODE_RANK:-0}"
hyper_rl_nproc="${NPROC_PER_NODE:-2}"
hyper_rl_master_addr="${MASTER_ADDR:-127.0.0.1}"
hyper_rl_master_port="${MASTER_PORT:-29500}"

if [[ ! -f "${hyper_rl_cann_env}" ]]; then
    echo "CANN environment script not found: ${hyper_rl_cann_env}" >&2
    exit 1
fi
if ! command -v "${hyper_rl_torchrun}" >/dev/null 2>&1; then
    echo "torchrun executable not found: ${hyper_rl_torchrun}" >&2
    exit 1
fi
if [[ ! -f "${hyper_rl_config}" ]]; then
    echo "Hyper-RL configuration not found: ${hyper_rl_config}" >&2
    exit 1
fi
if [[ ! "${hyper_rl_nproc}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer: ${hyper_rl_nproc}" >&2
    exit 1
fi

# Some CANN releases return a non-zero status when an optional driver metadata
# probe is absent, even though the required environment variables were set.
# Keep that probe from tripping this script's errexit; torch_npu/torchrun will
# still fail with an actionable error if the runtime is genuinely unavailable.
# shellcheck disable=SC1090
if ! source "${hyper_rl_cann_env}"; then
    echo "Warning: CANN environment script returned a non-zero status: ${hyper_rl_cann_env}" >&2
fi

export HYPER_PARALLEL_PLATFORM=torch
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${hyper_rl_project_root}:${hyper_rl_repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Hyper-Parallel root: ${hyper_rl_repo_root}"
echo "Config: ${hyper_rl_config}"
echo "Visible NPU devices: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Distributed launch: nnodes=${hyper_rl_nnodes}, node_rank=${hyper_rl_node_rank}, nproc_per_node=${hyper_rl_nproc}"

cd "${hyper_rl_repo_root}"
exec "${hyper_rl_torchrun}" \
    --master_addr="${hyper_rl_master_addr}" \
    --master_port="${hyper_rl_master_port}" \
    --nnodes="${hyper_rl_nnodes}" \
    --node_rank="${hyper_rl_node_rank}" \
    --nproc_per_node="${hyper_rl_nproc}" \
    hyper_parallel/rl/examples/train_rl.py \
    "${hyper_rl_config}" \
    "$@"
