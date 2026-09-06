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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../.." && pwd)
workspace_root=$(cd -- "${repo_root}/.." && pwd)

deployment=${1:-colocated}
[[ "${deployment}" == "colocated" || "${deployment}" == "disjoint" ]] || {
    printf 'Usage: %s [colocated|disjoint]\n' "$0" >&2
    exit 1
}

image=${HYPER_QWEN3_IMAGE:-swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64}
model_root=${HYPER_QWEN3_MODEL_ROOT:-${workspace_root}/models/Qwen3-4B}
data_root=${HYPER_QWEN3_DATA_ROOT:-${workspace_root}/data/gsm8k}
result_root=${HYPER_QWEN3_RESULT_ROOT:-${repo_root}/.rollout-results/qwen3-consistency-smoke}
visible_devices=${HYPER_QWEN3_VISIBLE_DEVICES-0,1}
tensor_parallel_size=${HYPER_QWEN3_TP:-1}
weight_sync_strategy=${HYPER_QWEN3_WEIGHT_SYNC_STRATEGY:-full_gather}
weight_sync_fallback=${HYPER_QWEN3_WEIGHT_SYNC_FALLBACK:-none}
trainer_count_override=${HYPER_QWEN3_TRAINER_COUNT:-}
rollout_dp_override=${HYPER_QWEN3_ROLLOUT_DP:-}
default_learning_rate=0
default_prompt_batch_size=2
default_rollout_port=8100
if [[ "${tensor_parallel_size}" == "2" ]]; then
    default_learning_rate=1e-6
    default_prompt_batch_size=1
    default_rollout_port=8422
fi
timeout_seconds=${HYPER_QWEN3_TIMEOUT_SECONDS:-3600}
max_steps=${HYPER_QWEN3_MAX_STEPS:-2}
learning_rate=${HYPER_QWEN3_LEARNING_RATE:-${default_learning_rate}}
learning_gate_enabled=${HYPER_QWEN3_LEARNING_GATE_ENABLED:-false}
default_log_name=qwen3-tp${tensor_parallel_size}-consistency.log
if [[ "${deployment}" == "disjoint" ]]; then
    default_log_name=qwen3-disjoint-tp${tensor_parallel_size}-consistency.log
fi
log_name=${HYPER_QWEN3_LOG_NAME:-${default_log_name}}
prompt_batch_size=${HYPER_QWEN3_PROMPT_BATCH_SIZE:-${default_prompt_batch_size}}
max_new_tokens=${HYPER_QWEN3_MAX_NEW_TOKENS:-32}
num_return_sequences=${HYPER_QWEN3_NUM_RETURN_SEQUENCES:-4}
rollout_port=${HYPER_QWEN3_ROLLOUT_PORT:-${default_rollout_port}}
dry_run=${HYPER_QWEN3_DRY_RUN:-false}
shm_size=${HYPER_QWEN3_SHM_SIZE:-64g}
hccl_if_base_port=${HCCL_IF_BASE_PORT:-62200}
hccl_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE:-62200-62300}

validate_hccl_ports() {
    local base_port=$1
    local socket_range=$2
    local range_start
    local range_end
    [[ "${base_port}" =~ ^[0-9]{1,5}$ ]] || return 1
    [[ "${socket_range}" =~ ^([0-9]{1,5})-([0-9]{1,5})$ ]] || return 1
    range_start=${BASH_REMATCH[1]}
    range_end=${BASH_REMATCH[2]}
    (( 10#${range_start} >= 1024 &&
       10#${range_start} <= 10#${base_port} &&
       10#${base_port} <= 10#${range_end} &&
       10#${range_end} <= 65520 ))
}

validate_hccl_ports "${hccl_if_base_port}" "${hccl_socket_port_range}" || {
    printf 'HCCL ports must use START-END in [1024, 65520] and contain HCCL_IF_BASE_PORT; got base=%s range=%s\n' \
        "${hccl_if_base_port}" "${hccl_socket_port_range}" >&2
    exit 1
}

[[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    printf 'HYPER_QWEN3_VISIBLE_DEVICES must be a comma-separated NPU list, got: %s\n' \
        "${visible_devices}" >&2
    exit 1
}
IFS=',' read -r -a device_ids <<< "${visible_devices}"
declare -A seen_devices=()
for device_id in "${device_ids[@]}"; do
    normalized_device_id=$((10#${device_id}))
    [[ -z "${seen_devices[${normalized_device_id}]+present}" ]] || {
        printf 'HYPER_QWEN3_VISIBLE_DEVICES must contain distinct NPUs, got: %s\n' \
            "${visible_devices}" >&2
        exit 1
    }
    seen_devices[${normalized_device_id}]=1
done
device_count=${#device_ids[@]}
[[ "${tensor_parallel_size}" == "1" || "${tensor_parallel_size}" == "2" ]] || {
    printf 'HYPER_QWEN3_TP must be 1 or 2, got: %s\n' "${tensor_parallel_size}" >&2
    exit 1
}
[[ "${weight_sync_strategy}" == "full_gather" || "${weight_sync_strategy}" == "direct_reshard" ]] || {
    printf 'HYPER_QWEN3_WEIGHT_SYNC_STRATEGY must be full_gather or direct_reshard\n' >&2
    exit 1
}
[[ "${weight_sync_fallback}" == "none" || "${weight_sync_fallback}" == "full_gather" ]] || {
    printf 'HYPER_QWEN3_WEIGHT_SYNC_FALLBACK must be none or full_gather\n' >&2
    exit 1
}
[[ -z "${trainer_count_override}" || "${trainer_count_override}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_TRAINER_COUNT must be a positive integer\n' >&2
    exit 1
}
[[ -z "${rollout_dp_override}" || "${rollout_dp_override}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_ROLLOUT_DP must be a positive integer\n' >&2
    exit 1
}
if [[ "${weight_sync_strategy}" != "direct_reshard" && "${weight_sync_fallback}" != "none" ]]; then
    printf 'HYPER_QWEN3_WEIGHT_SYNC_FALLBACK requires direct_reshard\n' >&2
    exit 1
fi
[[ -z "${HYPER_QWEN3_VLLM_TOPOLOGY:-}" ]] || {
    printf 'HYPER_QWEN3_VLLM_TOPOLOGY topology option was removed\n' >&2
    exit 1
}
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_MAX_STEPS must be positive, got: %s\n' "${max_steps}" >&2
    exit 1
}
[[ "${learning_rate}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || {
    printf 'HYPER_QWEN3_LEARNING_RATE must be non-negative, got: %s\n' "${learning_rate}" >&2
    exit 1
}
[[ "${learning_gate_enabled}" == "true" || "${learning_gate_enabled}" == "false" ]] || {
    printf 'HYPER_QWEN3_LEARNING_GATE_ENABLED must be true or false, got: %s\n' \
        "${learning_gate_enabled}" >&2
    exit 1
}
[[ "${prompt_batch_size}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_PROMPT_BATCH_SIZE must be positive, got: %s\n' \
        "${prompt_batch_size}" >&2
    exit 1
}
[[ "${max_new_tokens}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_MAX_NEW_TOKENS must be positive, got: %s\n' \
        "${max_new_tokens}" >&2
    exit 1
}
[[ "${num_return_sequences}" =~ ^[1-9][0-9]*$ ]] && (( num_return_sequences >= 2 )) || {
    printf 'HYPER_QWEN3_NUM_RETURN_SEQUENCES must be an integer of at least 2, got: %s\n' \
        "${num_return_sequences}" >&2
    exit 1
}
[[ "${rollout_port}" =~ ^[1-9][0-9]*$ ]] && (( rollout_port <= 65535 )) || {
    printf 'HYPER_QWEN3_ROLLOUT_PORT must be an integer from 1 to 65535, got: %s\n' \
        "${rollout_port}" >&2
    exit 1
}
[[ "${dry_run}" == "true" || "${dry_run}" == "false" ]] || {
    printf 'HYPER_QWEN3_DRY_RUN must be true or false, got: %s\n' "${dry_run}" >&2
    exit 1
}
[[ "${shm_size}" =~ ^[1-9][0-9]*[gGmM]$ ]] || {
    printf 'HYPER_QWEN3_SHM_SIZE must be a positive Docker size such as 64g\n' >&2
    exit 1
}
trainer_count=${device_count}
training_device_ids=("${device_ids[@]}")
rollout_device_ids=("${device_ids[@]}")
if [[ "${deployment}" == "colocated" ]]; then
    if (( device_count < 2 )); then
        printf 'internal_dp requires at least two visible NPUs, got: %s\n' \
            "${visible_devices}" >&2
        exit 1
    fi
    if (( device_count % tensor_parallel_size != 0 )); then
        printf 'Device count %s must be divisible by TP%s\n' \
            "${device_count}" "${tensor_parallel_size}" >&2
        exit 1
    fi
    if (( tensor_parallel_size == 2 && device_count != 4 )); then
        printf 'Qwen3 colocated TP2 consistency requires the validated four-device topology\n' >&2
        exit 1
    fi
    if [[ -n "${trainer_count_override}" && "${trainer_count_override}" != "${device_count}" ]]; then
        printf 'Colocated Trainer count %s must equal selected device count %s\n' \
            "${trainer_count_override}" "${device_count}" >&2
        exit 1
    fi
    rollout_data_parallel_size=$((trainer_count / tensor_parallel_size))
    if [[ -n "${rollout_dp_override}" && "${rollout_dp_override}" != "${rollout_data_parallel_size}" ]]; then
        printf 'Colocated rollout DP%s must equal Trainer world/TP=%s\n' \
            "${rollout_dp_override}" "${rollout_data_parallel_size}" >&2
        exit 1
    fi
else
    trainer_count=${trainer_count_override:-2}
    if (( trainer_count >= device_count )); then
        printf 'Disjoint deployment requires rollout NPUs after %s Trainer devices; got %s total devices\n' \
            "${trainer_count}" "${device_count}" >&2
        exit 1
    fi
    if (( trainer_count % tensor_parallel_size != 0 )); then
        printf 'Disjoint Trainer count %s must be divisible by TP%s\n' \
            "${trainer_count}" "${tensor_parallel_size}" >&2
        exit 1
    fi
    rollout_device_count=$((device_count - trainer_count))
    if (( rollout_device_count % tensor_parallel_size != 0 )); then
        printf 'Disjoint rollout device count %s must be divisible by TP%s\n' \
            "${rollout_device_count}" "${tensor_parallel_size}" >&2
        exit 1
    fi
    rollout_data_parallel_size=$((rollout_device_count / tensor_parallel_size))
    if [[ -n "${rollout_dp_override}" && "${rollout_dp_override}" != "${rollout_data_parallel_size}" ]]; then
        printf 'Disjoint rollout DP%s requires %s rollout devices for TP%s, got %s\n' \
            "${rollout_dp_override}" "$((rollout_dp_override * tensor_parallel_size))" \
            "${tensor_parallel_size}" "${rollout_device_count}" >&2
        exit 1
    fi
    training_device_ids=("${device_ids[@]:0:trainer_count}")
    rollout_device_ids=("${device_ids[@]:trainer_count}")
fi
[[ "${log_name}" =~ ^[A-Za-z0-9._-]+[.]log$ ]] || {
    printf 'HYPER_QWEN3_LOG_NAME must be a plain .log filename, got: %s\n' "${log_name}" >&2
    exit 1
}
trainer_dp_shard=$((trainer_count / tensor_parallel_size))
training_visible=$(IFS=','; printf '%s' "${training_device_ids[*]}")
rollout_visible=$(IFS=','; printf '%s' "${rollout_device_ids[*]}")
global_prompt_count=$((trainer_dp_shard * prompt_batch_size))
max_train_samples=${HYPER_QWEN3_MAX_TRAIN_SAMPLES:-${global_prompt_count}}
[[ "${max_train_samples}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_MAX_TRAIN_SAMPLES must be positive, got: %s\n' \
        "${max_train_samples}" >&2
    exit 1
}
(( max_train_samples >= global_prompt_count )) || {
    printf 'HYPER_QWEN3_MAX_TRAIN_SAMPLES must be at least the global prompt count %s\n' \
        "${global_prompt_count}" >&2
    exit 1
}
response_mini_batch_size=$((prompt_batch_size * num_return_sequences))
config_name=qwen3_4b_gsm8k_vllm_production.yaml
if (( tensor_parallel_size == 2 )); then
    config_name=qwen3_4b_gsm8k_vllm_tp2_consistency.yaml
fi
printf '%s\n' \
    "image=${image}" \
    "deployment=${deployment}" \
    "visible_devices=${visible_devices}" \
    "device_count=${device_count}" \
    "trainer_visible_devices=${training_visible}" \
    "rollout_visible_devices=${rollout_visible}" \
    "trainer_world_size=${trainer_count}" \
    "trainer_dp_shard=${trainer_dp_shard}" \
    "trainer_tensor_parallel_size=${tensor_parallel_size}" \
    "rollout_tensor_parallel_size=${tensor_parallel_size}" \
    "rollout_data_parallel_size=${rollout_data_parallel_size}" \
    "weight_sync_strategy=${weight_sync_strategy}" \
    "weight_sync_fallback=${weight_sync_fallback}" \
    "rollout_port=${rollout_port}" \
    "api_server_count=auto" \
    "prompt_batch_size=${prompt_batch_size}" \
    "global_prompt_count=${global_prompt_count}" \
    "num_return_sequences=${num_return_sequences}" \
    "global_child_count=$((global_prompt_count * num_return_sequences))" \
    "response_mini_batch_size=${response_mini_batch_size}" \
    "max_train_samples=${max_train_samples}" \
    "max_new_tokens=${max_new_tokens}" \
    "max_steps=${max_steps}" \
    "learning_rate=${learning_rate}" \
    "learning_gate_enabled=${learning_gate_enabled}" \
    "max_num_batched_tokens=2048" \
    "max_num_seqs=auto" \
    "config_name=${config_name}" \
    "torchrun_nproc_per_node=${trainer_count}" \
    "override_train_accelerator_dp_shard=${trainer_dp_shard}" \
    "override_train_accelerator_tp=${tensor_parallel_size}" \
    "override_rollout_data_parallel_size=${rollout_data_parallel_size}" \
    "override_rollout_tensor_parallel_size=${tensor_parallel_size}" \
    "override_max_num_seqs=omitted"
if [[ "${dry_run}" == "true" ]]; then
    exit 0
fi

[[ -d "${model_root}" ]] || {
    printf 'Qwen3 model directory is unavailable: %s\n' "${model_root}" >&2
    exit 1
}
[[ -f "${data_root}/train.parquet" && -f "${data_root}/test.parquet" ]] || {
    printf 'GSM8K train.parquet and test.parquet are required under: %s\n' "${data_root}" >&2
    exit 1
}

mkdir -p "${result_root}"

docker run --rm --privileged --shm-size="${shm_size}" --network=host \
    -e "ASCEND_RT_VISIBLE_DEVICES=${visible_devices}" \
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e "HCCL_IF_BASE_PORT=${hccl_if_base_port}" \
    -e "HCCL_NPU_SOCKET_PORT_RANGE=${hccl_socket_port_range}" \
    -e "HYPER_RUN_TIMEOUT_SECONDS=${timeout_seconds}" \
    -e "HYPER_QWEN3_MAX_STEPS=${max_steps}" \
    -e "HYPER_QWEN3_LEARNING_RATE=${learning_rate}" \
    -e "HYPER_QWEN3_LEARNING_GATE_ENABLED=${learning_gate_enabled}" \
    -e "HYPER_QWEN3_LOG_NAME=${log_name}" \
    -e "HYPER_QWEN3_DEPLOYMENT=${deployment}" \
    -e "HYPER_QWEN3_TRAINER_COUNT=${trainer_count}" \
    -e "HYPER_QWEN3_ROLLOUT_VISIBLE_DEVICES=${rollout_visible}" \
    -e "HYPER_QWEN3_TRAINER_DP_SHARD=${trainer_dp_shard}" \
    -e "HYPER_QWEN3_TENSOR_PARALLEL_SIZE=${tensor_parallel_size}" \
    -e "HYPER_QWEN3_WEIGHT_SYNC_STRATEGY=${weight_sync_strategy}" \
    -e "HYPER_QWEN3_WEIGHT_SYNC_FALLBACK=${weight_sync_fallback}" \
    -e "HYPER_QWEN3_ROLLOUT_DATA_PARALLEL_SIZE=${rollout_data_parallel_size}" \
    -e "HYPER_QWEN3_ROLLOUT_PORT=${rollout_port}" \
    -e "HYPER_QWEN3_PROMPT_BATCH_SIZE=${prompt_batch_size}" \
    -e "HYPER_QWEN3_GLOBAL_PROMPT_COUNT=${global_prompt_count}" \
    -e "HYPER_QWEN3_MAX_TRAIN_SAMPLES=${max_train_samples}" \
    -e "HYPER_QWEN3_MAX_NEW_TOKENS=${max_new_tokens}" \
    -e "HYPER_QWEN3_RESPONSE_MINI_BATCH_SIZE=${response_mini_batch_size}" \
    -e "HYPER_QWEN3_NUM_RETURN_SEQUENCES=${num_return_sequences}" \
    -e "HYPER_QWEN3_CONFIG_NAME=${config_name}" \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${repo_root}:/workspace/hyper-parallel:ro" \
    -v "${model_root}:/models/Qwen3-4B:ro" \
    -v "${data_root}:/data/gsm8k:ro" \
    -v "${result_root}:/results" \
    -w /workspace/hyper-parallel \
    "${image}" /bin/bash -lc '
        set -euo pipefail
        unset VLLM_PLUGINS
        export PYTHONPATH=/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel:${PYTHONPATH:-}

        patch_file=/workspace/hyper-parallel/hyper_parallel/rl/examples/patches/vllm-dp-coordinator-timeout.patch
        if patch --dry-run -d /vllm-workspace/vllm -p1 < "${patch_file}" >/dev/null; then
            patch -d /vllm-workspace/vllm -p1 < "${patch_file}" >/dev/null
        elif patch --dry-run -R -d /vllm-workspace/vllm -p1 < "${patch_file}" >/dev/null; then
            printf "vLLM startup patch is already applied\n"
        else
            printf "vLLM startup patch is incompatible with the installed vLLM source\n" >&2
            exit 1
        fi

        set +e
        rollout_args=(
            --consistency.enabled=true
            --rollout.vllm.deployment="${HYPER_QWEN3_DEPLOYMENT}"
            --rollout.vllm.data_parallel_size="${HYPER_QWEN3_ROLLOUT_DATA_PARALLEL_SIZE}"
            --rollout.vllm.tensor_parallel_size="${HYPER_QWEN3_TENSOR_PARALLEL_SIZE}"
            --rollout.vllm.weight_sync.strategy="${HYPER_QWEN3_WEIGHT_SYNC_STRATEGY}"
            --rollout.vllm.weight_sync.fallback_strategy="${HYPER_QWEN3_WEIGHT_SYNC_FALLBACK}"
            --rollout.vllm.port="${HYPER_QWEN3_ROLLOUT_PORT}"
            --data.max_train_samples="${HYPER_QWEN3_MAX_TRAIN_SAMPLES}"
            --data.shuffle=false
            --rollout.num_return_sequences="${HYPER_QWEN3_NUM_RETURN_SEQUENCES}"
            --rollout.max_new_tokens="${HYPER_QWEN3_MAX_NEW_TOKENS}"
            --train.prompt_batch_size="${HYPER_QWEN3_PROMPT_BATCH_SIZE}"
            --train.response_mini_batch_size="${HYPER_QWEN3_RESPONSE_MINI_BATCH_SIZE}"
            --train.accelerator.dp_shard="${HYPER_QWEN3_TRAINER_DP_SHARD}"
            --train.accelerator.tp="${HYPER_QWEN3_TENSOR_PARALLEL_SIZE}"
            --evaluation.enabled=false
            --train.checkpoint.save_steps=0
            --train.checkpoint.save_final=false
            --train.checkpoint.verify_reload=false
            --logging.backends='[console]'
            --logging.wandb.mode=disabled
        )
        if [[ "${HYPER_QWEN3_DEPLOYMENT}" == "disjoint" ]]; then
            rollout_args+=(
                --rollout.vllm.visible_devices="${HYPER_QWEN3_ROLLOUT_VISIBLE_DEVICES}"
            )
        fi
        {
            printf "resolved_world_size=%s resolved_prompt_batch_size=%s resolved_global_prompts=%s\n" \
                "${HYPER_QWEN3_TRAINER_COUNT}" \
                "${HYPER_QWEN3_PROMPT_BATCH_SIZE}" \
                "${HYPER_QWEN3_GLOBAL_PROMPT_COUNT}"
            timeout --signal=TERM --kill-after=30s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
                python -m torch.distributed.run --standalone \
                --nproc_per_node="${HYPER_QWEN3_TRAINER_COUNT}" \
                /workspace/hyper-parallel/hyper_parallel/rl/examples/train_rl.py \
                "/workspace/hyper-parallel/hyper_parallel/rl/examples/configs/${HYPER_QWEN3_CONFIG_NAME}" \
                --train.max_steps="${HYPER_QWEN3_MAX_STEPS}" \
                --train.optimizer.lr="${HYPER_QWEN3_LEARNING_RATE}" \
                --train.learning_gate.enabled="${HYPER_QWEN3_LEARNING_GATE_ENABLED}" \
                "${rollout_args[@]}"
        } 2>&1 | tee "/results/${HYPER_QWEN3_LOG_NAME}"
        status="${PIPESTATUS[0]}"
        set -e
        exit "${status}"
    '
