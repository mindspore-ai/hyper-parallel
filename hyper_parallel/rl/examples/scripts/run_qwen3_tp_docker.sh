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

deployment=${1:-}
[[ "${deployment}" == "colocated" || "${deployment}" == "disjoint" ]] || {
    printf 'Usage: %s {colocated|disjoint}\n' "$0" >&2
    exit 1
}

implementation=${HYPER_QWEN3_TP_IMPLEMENTATION:-hyper}
[[ "${implementation}" == "hyper" || "${implementation}" == "native" ]] || {
    printf 'HYPER_QWEN3_TP_IMPLEMENTATION must be hyper or native\n' >&2
    exit 1
}
trainer_tp=${HYPER_QWEN3_TP_TRAINER_TP:-1}
rollout_tp=${HYPER_QWEN3_TP_ROLLOUT_TP:-2}
[[ "${trainer_tp}" == "1" || "${trainer_tp}" == "2" ]] || {
    printf 'HYPER_QWEN3_TP_TRAINER_TP must be 1 or 2\n' >&2
    exit 1
}
[[ "${rollout_tp}" == "1" || "${rollout_tp}" == "2" ]] || {
    printf 'HYPER_QWEN3_TP_ROLLOUT_TP must be 1 or 2\n' >&2
    exit 1
}

weight_sync_strategy=${HYPER_QWEN3_TP_WEIGHT_SYNC_STRATEGY:-direct_reshard}
[[ "${weight_sync_strategy}" == "direct_reshard" || "${weight_sync_strategy}" == "full_gather" ]] || {
    printf 'HYPER_QWEN3_TP_WEIGHT_SYNC_STRATEGY must be direct_reshard or full_gather\n' >&2
    exit 1
}
fallback_strategy=${HYPER_QWEN3_TP_WEIGHT_SYNC_FALLBACK:-full_gather}
[[ "${fallback_strategy}" == "full_gather" || "${fallback_strategy}" == "none" ]] || {
    printf 'HYPER_QWEN3_TP_WEIGHT_SYNC_FALLBACK must be full_gather or none\n' >&2
    exit 1
}
max_steps=${HYPER_QWEN3_TP_MAX_STEPS:-1}
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_TP_MAX_STEPS must be a positive integer\n' >&2
    exit 1
}
num_return_sequences=${HYPER_QWEN3_TP_NUM_RETURN_SEQUENCES:-2}
max_new_tokens=${HYPER_QWEN3_TP_MAX_NEW_TOKENS:-32}
max_model_len=${HYPER_QWEN3_TP_MAX_MODEL_LEN:-256}
max_num_seqs=${HYPER_QWEN3_TP_MAX_NUM_SEQS:-2}
max_num_batched_tokens=${HYPER_QWEN3_TP_MAX_NUM_BATCHED_TOKENS:-1024}
learning_rate=${HYPER_QWEN3_TP_LEARNING_RATE:-1e-6}
rollout_seed=${HYPER_QWEN3_TP_ROLLOUT_SEED:-}
trainer_count_override=${HYPER_QWEN3_TP_TRAINER_COUNT:-}
rollout_dp_override=${HYPER_QWEN3_TP_ROLLOUT_DP:-}
rollout_port_override=${HYPER_QWEN3_TP_ROLLOUT_PORT:-}
for positive_value in \
    "${num_return_sequences}" "${max_new_tokens}" "${max_model_len}" \
    "${max_num_seqs}" "${max_num_batched_tokens}"; do
    [[ "${positive_value}" =~ ^[1-9][0-9]*$ ]] || {
        printf 'Qwen3 TP workload sizes must be positive integers, got: %s\n' "${positive_value}" >&2
        exit 1
    }
done
[[ "${learning_rate}" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || {
    printf 'HYPER_QWEN3_TP_LEARNING_RATE must be non-negative, got: %s\n' "${learning_rate}" >&2
    exit 1
}
[[ -z "${rollout_seed}" || "${rollout_seed}" =~ ^[0-9]+$ ]] || {
    printf 'HYPER_QWEN3_TP_ROLLOUT_SEED must be a non-negative integer, got: %s\n' "${rollout_seed}" >&2
    exit 1
}
[[ -z "${trainer_count_override}" || "${trainer_count_override}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_TP_TRAINER_COUNT must be a positive integer\n' >&2
    exit 1
}
[[ -z "${rollout_dp_override}" || "${rollout_dp_override}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_QWEN3_TP_ROLLOUT_DP must be a positive integer\n' >&2
    exit 1
}
[[ -z "${rollout_port_override}" || "${rollout_port_override}" =~ ^[1-9][0-9]*$ ]] \
    && [[ -z "${rollout_port_override}" || "${rollout_port_override}" -le 65535 ]] || {
    printf 'HYPER_QWEN3_TP_ROLLOUT_PORT must be an integer from 1 to 65535\n' >&2
    exit 1
}
[[ -z "${HYPER_QWEN3_TP_TOPOLOGY:-}" ]] || {
    printf 'HYPER_QWEN3_TP_TOPOLOGY topology option was removed\n' >&2
    exit 1
}

workspace_root=$(cd -- "${repo_root}/.." && pwd)
image=${HYPER_QWEN3_TP_IMAGE:-swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64}
model_root=${HYPER_QWEN3_TP_MODEL_ROOT:-${workspace_root}/models/Qwen3-4B}
data_root=${HYPER_QWEN3_TP_DATA_ROOT:-${workspace_root}/data/gsm8k}
result_root=${HYPER_QWEN3_TP_RESULT_ROOT:-${repo_root}/.rollout-results/qwen3-tp-smoke}
requested_devices=${HYPER_QWEN3_TP_VISIBLE_DEVICES:-}
if [[ -n "${requested_devices}" && ! "${requested_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'Invalid HYPER_QWEN3_TP_VISIBLE_DEVICES: %s\n' "${requested_devices}" >&2
    exit 1
fi
timeout_seconds=${HYPER_QWEN3_TP_TIMEOUT_SECONDS:-3600}
shm_size=${HYPER_QWEN3_TP_SHM_SIZE:-64g}
hccl_if_base_port=${HCCL_IF_BASE_PORT:-62400}
hccl_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE:-62400-62550}
[[ "${shm_size}" =~ ^[1-9][0-9]*[gGmM]$ ]] || {
    printf 'HYPER_QWEN3_TP_SHM_SIZE must be a positive Docker size such as 64g\n' >&2
    exit 1
}

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
default_vllm_startup_patch=${repo_root}/hyper_parallel/rl/examples/patches/vllm-dp-coordinator-timeout.patch
vllm_startup_patch=${HYPER_QWEN3_TP_VLLM_STARTUP_PATCH-${default_vllm_startup_patch}}
vllm_patch_mount=()
if [[ -n "${vllm_startup_patch}" ]]; then
    [[ -f "${vllm_startup_patch}" ]] || {
        printf 'HYPER_QWEN3_TP_VLLM_STARTUP_PATCH is not a file: %s\n' \
            "${vllm_startup_patch}" >&2
        exit 1
    }
    vllm_startup_patch=$(cd -- "$(dirname -- "${vllm_startup_patch}")" && pwd)/$(basename -- "${vllm_startup_patch}")
    vllm_patch_mount=(-v "${vllm_startup_patch}:/patches/vllm-startup.patch:ro")
fi
dry_run=${HYPER_QWEN3_TP_DRY_RUN:-false}
[[ "${dry_run}" == "true" || "${dry_run}" == "false" ]] || {
    printf 'HYPER_QWEN3_TP_DRY_RUN must be true or false\n' >&2
    exit 1
}
trainer_count=${trainer_count_override:-2}
rollout_dp=1
required_devices=2
rollout_port=8300
gpu_memory_utilization=0.30
if [[ "${deployment}" == "colocated" ]]; then
    if [[ -n "${requested_devices}" ]]; then
        IFS=',' read -r -a requested_device_list <<<"${requested_devices}"
        inferred_trainer_count=${#requested_device_list[@]}
        if [[ -n "${trainer_count_override}" && "${trainer_count_override}" != "${inferred_trainer_count}" ]]; then
            printf 'Colocated Trainer count %s must equal selected device count %s\n' \
                "${trainer_count_override}" "${inferred_trainer_count}" >&2
            exit 1
        fi
        trainer_count=${inferred_trainer_count}
    fi
    if (( trainer_count % rollout_tp != 0 )); then
        printf 'Colocated rollout TP%s requires world size %s to be divisible by TP\n' \
            "${rollout_tp}" "${trainer_count}" >&2
        exit 1
    fi
    rollout_dp=$((trainer_count / rollout_tp))
    if [[ -n "${rollout_dp_override}" && "${rollout_dp_override}" != "${rollout_dp}" ]]; then
        printf 'Colocated rollout DP%s must equal Trainer world/TP=%s\n' \
            "${rollout_dp_override}" "${rollout_dp}" >&2
        exit 1
    fi
    required_devices=${trainer_count}
    rollout_port=8500
    if (( rollout_dp > 1 )); then
        gpu_memory_utilization=0.25
    fi
elif [[ "${deployment}" == "disjoint" ]]; then
    rollout_dp=${rollout_dp_override:-2}
    required_devices=$((trainer_count + rollout_dp * rollout_tp))
    rollout_port=8400
    gpu_memory_utilization=0.50
fi
rollout_port=${rollout_port_override:-${rollout_port}}
rollout_device_count=$((rollout_dp * rollout_tp))
if (( trainer_count % trainer_tp != 0 )); then
    printf 'Trainer world size %s must be divisible by Trainer TP%s\n' \
        "${trainer_count}" "${trainer_tp}" >&2
    exit 1
fi
trainer_dp_shard=$((trainer_count / trainer_tp))
resolved_rollout_devices=auto
if [[ -n "${requested_devices}" ]]; then
    IFS=',' read -r -a requested_device_list <<<"${requested_devices}"
    [[ "${#requested_device_list[@]}" -eq "${required_devices}" ]] || {
        printf '%s rollout requires exactly %d NPUs, got: %s\n' \
            "${deployment}" "${required_devices}" "${requested_devices}" >&2
        exit 1
    }
    declare -A seen_requested_devices=()
    for requested_device in "${requested_device_list[@]}"; do
        normalized_requested_device=$((10#${requested_device}))
        [[ -z "${seen_requested_devices[${normalized_requested_device}]+present}" ]] || {
            printf 'Selected NPUs must be unique: %s\n' "${requested_devices}" >&2
            exit 1
        }
        seen_requested_devices[${normalized_requested_device}]=1
    done
    rollout_device_list=("${requested_device_list[@]}")
    if [[ "${deployment}" == "disjoint" ]]; then
        rollout_device_list=("${requested_device_list[@]:trainer_count}")
    fi
    resolved_rollout_devices=$(IFS=','; printf '%s' "${rollout_device_list[*]}")
fi
config_path=${repo_root}/hyper_parallel/rl/examples/configs/qwen3_4b_gsm8k_vllm_production.yaml
mkdir -p "${result_root}"
result_root=$(cd -- "${result_root}" && pwd)

if [[ "${dry_run}" == "true" ]]; then
    printf 'image=%s\n' "${image}"
    printf 'deployment=%s\n' "${deployment}"
    printf 'trainer_count=%s\n' "${trainer_count}"
    printf 'trainer_tensor_parallel_size=%s\n' "${trainer_tp}"
    printf 'trainer_dp_shard_size=%s\n' "${trainer_dp_shard}"
    printf 'rollout_data_parallel_size=%s\n' "${rollout_dp}"
    printf 'rollout_tensor_parallel_size=%s\n' "${rollout_tp}"
    printf 'rollout_device_count=%s\n' "${rollout_device_count}"
    printf 'rollout_visible_devices=%s\n' "${resolved_rollout_devices}"
    printf 'rollout_port=%s\n' "${rollout_port}"
    printf 'required_devices=%s\n' "${required_devices}"
    printf 'result_root=%s\n' "${result_root}"
    printf 'num_return_sequences=%s\n' "${num_return_sequences}"
    printf 'max_new_tokens=%s\n' "${max_new_tokens}"
    printf 'rollout_seed=%s\n' "${rollout_seed}"
    printf 'learning_rate=%s\n' "${learning_rate}"
    exit 0
fi

command -v docker >/dev/null 2>&1 || {
    printf 'docker executable is required\n' >&2
    exit 1
}
command -v npu-smi >/dev/null 2>&1 || {
    printf 'npu-smi is required for automatic NPU allocation\n' >&2
    exit 1
}
[[ -d "${model_root}" ]] || {
    printf 'Qwen3 model directory is unavailable: %s\n' "${model_root}" >&2
    exit 1
}
[[ -f "${data_root}/train.parquet" && -f "${data_root}/test.parquet" ]] || {
    printf 'GSM8K parquet files are unavailable under: %s\n' "${data_root}" >&2
    exit 1
}
npu_info=$(npu-smi info)
mapfile -t healthy_devices < <(
    awk -F'|' '
        $3 ~ /^[[:space:]]*OK[[:space:]]*$/ {
            field = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", field)
            split(field, values, /[[:space:]]+/)
            print values[1]
        }
    ' <<<"${npu_info}"
)
mapfile -t idle_devices < <(
    sed -n 's/.*No running processes found in NPU \([0-9][0-9]*\).*/\1/p' \
        <<<"${npu_info}"
)
available_devices=()
for device in "${healthy_devices[@]}"; do
    for idle_device in "${idle_devices[@]}"; do
        if [[ "${device}" == "${idle_device}" ]]; then
            available_devices+=("${device}")
            break
        fi
    done
done

if [[ -n "${requested_devices}" ]]; then
    IFS=',' read -r -a selected_devices <<<"${requested_devices}"
else
    selected_devices=("${available_devices[@]:0:required_devices}")
fi
[[ "${#selected_devices[@]}" -eq "${required_devices}" ]] || {
    printf '%s rollout requires %d healthy idle NPUs; available=%s selected=%s\n' \
        "${deployment}" "${required_devices}" "${available_devices[*]}" \
        "${selected_devices[*]}" >&2
    exit 1
}
[[ "$(printf '%s\n' "${selected_devices[@]}" | sort -u | wc -l)" -eq "${required_devices}" ]] || {
    printf 'Selected NPUs must be unique: %s\n' "${selected_devices[*]}" >&2
    exit 1
}

training_devices=("${selected_devices[@]:0:trainer_count}")
rollout_devices=()
if [[ "${deployment}" == "disjoint" ]]; then
    rollout_devices=("${selected_devices[@]:trainer_count}")
fi
all_visible=$(IFS=','; printf '%s' "${selected_devices[*]}")
training_visible=$(IFS=','; printf '%s' "${training_devices[*]}")
rollout_visible=$(IFS=','; printf '%s' "${rollout_devices[*]}")
printf 'Qwen3 TP smoke: deployment=%s implementation=%s trainer_tp=%s rollout_tp=%s weight_sync=%s fallback=%s steps=%s training=%s rollout=%s\n' \
    "${deployment}" "${implementation}" "${trainer_tp}" "${rollout_tp}" \
    "${weight_sync_strategy}" "${fallback_strategy}" \
    "${max_steps}" "${training_visible}" "${rollout_visible:-colocated}"

container_name="hyper-qwen3-${implementation}-trainer-tp${trainer_tp}-rollout-tp${rollout_tp}-$$"
docker run --rm --name "${container_name}" --privileged --shm-size="${shm_size}" --network=host \
    -e "ASCEND_RT_VISIBLE_DEVICES=${all_visible}" \
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e "HCCL_IF_BASE_PORT=${hccl_if_base_port}" \
    -e "HCCL_NPU_SOCKET_PORT_RANGE=${hccl_socket_port_range}" \
    -e "HYPER_RUN_DEPLOYMENT=${deployment}" \
    -e "HYPER_RUN_IMPLEMENTATION=${implementation}" \
    -e "HYPER_RUN_TRAINER_TP=${trainer_tp}" \
    -e "HYPER_RUN_WEIGHT_SYNC_STRATEGY=${weight_sync_strategy}" \
    -e "HYPER_RUN_WEIGHT_SYNC_FALLBACK=${fallback_strategy}" \
    -e "HYPER_RUN_MAX_STEPS=${max_steps}" \
    -e "HYPER_RUN_NUM_RETURN_SEQUENCES=${num_return_sequences}" \
    -e "HYPER_RUN_MAX_NEW_TOKENS=${max_new_tokens}" \
    -e "HYPER_RUN_MAX_MODEL_LEN=${max_model_len}" \
    -e "HYPER_RUN_MAX_NUM_SEQS=${max_num_seqs}" \
    -e "HYPER_RUN_MAX_NUM_BATCHED_TOKENS=${max_num_batched_tokens}" \
    -e "HYPER_RUN_LEARNING_RATE=${learning_rate}" \
    -e "HYPER_RUN_ROLLOUT_SEED=${rollout_seed}" \
    -e "HYPER_RUN_TRAINER_COUNT=${trainer_count}" \
    -e "HYPER_RUN_TRAINER_DP_SHARD=${trainer_dp_shard}" \
    -e "HYPER_RUN_ROLLOUT_DP=${rollout_dp}" \
    -e "HYPER_RUN_ROLLOUT_TP=${rollout_tp}" \
    -e "HYPER_RUN_ROLLOUT_DEVICES=${rollout_visible}" \
    -e "HYPER_RUN_ROLLOUT_PORT=${rollout_port}" \
    -e "HYPER_RUN_GPU_MEMORY_UTILIZATION=${gpu_memory_utilization}" \
    -e "HYPER_RUN_TIMEOUT=${timeout_seconds}" \
    "${vllm_patch_mount[@]}" \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${repo_root}:/workspace/hyper-parallel:ro" \
    -v "${config_path}:/run-config.yaml:ro" \
    -v "${model_root}:/models/Qwen3-4B:ro" \
    -v "${data_root}:/data/gsm8k:ro" \
    -v "${result_root}:/results" \
    -w /workspace/hyper-parallel \
    "${image}" /bin/bash -lc '
        set -euo pipefail
        unset VLLM_PLUGINS
        export PYTHONPATH=/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel:${PYTHONPATH:-}
        if [[ -f /patches/vllm-startup.patch ]]; then
            if patch --dry-run -d /vllm-workspace/vllm -p1 < /patches/vllm-startup.patch >/dev/null; then
                patch -d /vllm-workspace/vllm -p1 < /patches/vllm-startup.patch >/dev/null
            elif patch --dry-run -R -d /vllm-workspace/vllm -p1 < /patches/vllm-startup.patch >/dev/null; then
                printf "vLLM startup patch is already applied\n"
            else
                printf "vLLM startup patch is incompatible with the installed vLLM source\n" >&2
                exit 1
            fi
        fi

        args=(
            /run-config.yaml
            --consistency.enabled=false
            --data.train_path=/data/gsm8k/train.parquet
            --data.max_train_samples=4
            --data.shuffle=false
            "--rollout.num_return_sequences=${HYPER_RUN_NUM_RETURN_SEQUENCES}"
            "--rollout.max_new_tokens=${HYPER_RUN_MAX_NEW_TOKENS}"
            "--rollout.vllm.deployment=${HYPER_RUN_DEPLOYMENT}"
            "--rollout.vllm.data_parallel_size=${HYPER_RUN_ROLLOUT_DP}"
            "--rollout.vllm.model_implementation=${HYPER_RUN_IMPLEMENTATION}"
            "--rollout.vllm.tensor_parallel_size=${HYPER_RUN_ROLLOUT_TP}"
            "--rollout.vllm.port=${HYPER_RUN_ROLLOUT_PORT}"
            --rollout.vllm.batch_invariant=false
            --rollout.vllm.enable_prefix_caching=false
            --rollout.vllm.enable_chunked_prefill=false
            "--rollout.vllm.gpu_memory_utilization=${HYPER_RUN_GPU_MEMORY_UTILIZATION}"
            --rollout.vllm.kv_cache_memory_bytes=1073741824
            "--rollout.vllm.max_model_len=${HYPER_RUN_MAX_MODEL_LEN}"
            "--rollout.vllm.max_num_seqs=${HYPER_RUN_MAX_NUM_SEQS}"
            "--rollout.vllm.max_num_batched_tokens=${HYPER_RUN_MAX_NUM_BATCHED_TOKENS}"
            "--rollout.vllm.weight_sync.strategy=${HYPER_RUN_WEIGHT_SYNC_STRATEGY}"
            "--rollout.vllm.weight_sync.fallback_strategy=${HYPER_RUN_WEIGHT_SYNC_FALLBACK}"
            "--train.max_steps=${HYPER_RUN_MAX_STEPS}"
            --train.prompt_batch_size=1
            --train.micro_batch_size=1
            "--train.response_mini_batch_size=${HYPER_RUN_NUM_RETURN_SEQUENCES}"
            "--train.accelerator.dp_shard=${HYPER_RUN_TRAINER_DP_SHARD}"
            "--train.accelerator.tp=${HYPER_RUN_TRAINER_TP}"
            "--train.optimizer.lr=${HYPER_RUN_LEARNING_RATE}"
            --evaluation.enabled=false
            --train.checkpoint.save_steps=0
            --train.checkpoint.save_final=false
            --train.checkpoint.verify_reload=false
            --logging.backends='[console]'
            --logging.wandb.mode=disabled
        )
        if [[ "${HYPER_RUN_DEPLOYMENT}" == "disjoint" ]]; then
            args+=("--rollout.vllm.visible_devices=${HYPER_RUN_ROLLOUT_DEVICES}")
        fi
        if [[ -n "${HYPER_RUN_ROLLOUT_SEED}" ]]; then
            args+=("--rollout.seed=${HYPER_RUN_ROLLOUT_SEED}")
        fi
        log_file="/results/${HYPER_RUN_DEPLOYMENT}-${HYPER_RUN_IMPLEMENTATION}-trainer-tp${HYPER_RUN_TRAINER_TP}-rollout-tp${HYPER_RUN_ROLLOUT_TP}-${HYPER_RUN_WEIGHT_SYNC_STRATEGY}.log"
        timeout --signal=TERM --kill-after=60s "${HYPER_RUN_TIMEOUT}s" \
            python -m torch.distributed.run --standalone --nproc_per_node="${HYPER_RUN_TRAINER_COUNT}" \
            hyper_parallel/rl/examples/train_rl.py "${args[@]}" 2>&1 | tee "${log_file}"
        exit "${PIPESTATUS[0]}"
    '
