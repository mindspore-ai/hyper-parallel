#!/usr/bin/env bash

set -euo pipefail

hyper_rl_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
hyper_rl_example_root="$(cd -- "${hyper_rl_script_dir}/.." && pwd)"
hyper_rl_project_root="$(cd -- "${hyper_rl_example_root}/.." && pwd)"
hyper_rl_repo_root="$(cd -- "${hyper_rl_project_root}/../.." && pwd)"
hyper_rl_workspace="$(cd -- "${hyper_rl_repo_root}/.." && pwd)"
hyper_rl_repo_name="$(basename -- "${hyper_rl_repo_root}")"
hyper_rl_container_root="/workspace/${hyper_rl_repo_name}"
hyper_rl_container_project_root="${hyper_rl_container_root}/hyper_parallel/rl"

hyper_rl_image="${HYPER_RL_IMAGE:-slime-small-test-preserved:20260720}"
hyper_rl_device_0="${NPU_DEVICE_0:-/dev/davinci4}"
hyper_rl_device_1="${NPU_DEVICE_1:-/dev/davinci5}"
hyper_rl_config="${HYPER_RL_CONFIG:-${hyper_rl_container_project_root}/examples/configs/qwen3_5_0_8b_gsm8k.yaml}"
hyper_rl_master_port="${MASTER_PORT:-29500}"
hyper_rl_hccl_socket_port_range="${HCCL_NPU_SOCKET_PORT_RANGE:-}"
hyper_rl_default_torchdata="/home/miniconda3/envs/twx_qwen3_bench_py311/lib/python3.11/site-packages/torchdata"
hyper_rl_torchdata="${TORCHDATA_PATH:-${hyper_rl_default_torchdata}}"

hyper_rl_devices=(
    "${hyper_rl_device_0}"
    "${hyper_rl_device_1}"
    /dev/davinci_manager
    /dev/devmm_svm
    /dev/hisi_hdc
)
for hyper_rl_required in "${hyper_rl_devices[@]}"; do
    if [[ ! -e "${hyper_rl_required}" ]]; then
        echo "Required Ascend device is missing: ${hyper_rl_required}" >&2
        exit 1
    fi
done

hyper_rl_mounts=(
    -v "${hyper_rl_workspace}:/workspace"
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
)
if [[ -d "${hyper_rl_torchdata}" ]]; then
    hyper_rl_mounts+=(
        -v "${hyper_rl_torchdata}:/usr/local/python3.11.15/lib/python3.11/site-packages/torchdata:ro"
    )
fi
if [[ -n "${hyper_rl_hccl_socket_port_range}" ]]; then
    hyper_rl_mounts+=(
        -e HCCL_NPU_SOCKET_PORT_RANGE="${hyper_rl_hccl_socket_port_range}"
    )
fi

exec docker run --rm --network none \
    --device="${hyper_rl_device_0}" \
    --device="${hyper_rl_device_1}" \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    "${hyper_rl_mounts[@]}" \
    -w "${hyper_rl_container_root}" \
    -e PYTHONPATH="${hyper_rl_container_project_root}:${hyper_rl_container_root}" \
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
    -e TORCHRUN_BIN=/usr/local/python3.11.15/bin/torchrun \
    -e HYPER_RL_CONFIG="${hyper_rl_config}" \
    -e MASTER_PORT="${hyper_rl_master_port}" \
    "${hyper_rl_image}" \
    bash hyper_parallel/rl/examples/scripts/run_qwen3_5_0_8b_gsm8k.sh "$@"
