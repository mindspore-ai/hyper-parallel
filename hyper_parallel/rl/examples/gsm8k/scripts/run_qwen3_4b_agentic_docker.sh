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
repo_root=$(cd -- "${script_dir}/../../../../.." && pwd)
workspace_root=$(cd -- "${repo_root}/.." && pwd)
default_image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64

: "${HYPER_VLLM_IMAGE:=${default_image}}"
: "${HYPER_CODEX_IMAGE:=${default_image}}"
: "${HYPER_VLLM_MODEL_ROOT:=${workspace_root}/models/Qwen3-4B}"
: "${HYPER_GSM8K_DATA_ROOT:=${workspace_root}/data/gsm8k}"
# Supported tasks: gsm8k, gsm8k_multi, gsm8k_codex, gsm8k_deepseek.
: "${HYPER_AGENTIC_TASK:=gsm8k}"
: "${HYPER_VLLM_VISIBLE_DEVICES:=0,1}"
: "${HYPER_VLLM_MODEL_IMPLEMENTATION:=native}"
: "${HYPER_VLLM_TIMEOUT_SECONDS:=3600}"
: "${HYPER_REQUIRE_LEARNING_UPDATE:=false}"
: "${HYPER_VLLM_PORT:=8100}"
: "${HYPER_CODEX_GATEWAY_PORT:=8200}"

case "${HYPER_AGENTIC_TASK}" in
    gsm8k|gsm8k_multi)
        runtime_image=${HYPER_VLLM_IMAGE}
        task_data_root=${HYPER_GSM8K_DATA_ROOT}
        container_data_root=/data/gsm8k
        config_relative=hyper_parallel/rl/examples/gsm8k/configs/qwen3_4b_gsm8k_vllm_production.yaml
        required_data_files=(train.parquet)
        ;;
    gsm8k_codex)
        runtime_image=${HYPER_CODEX_IMAGE}
        task_data_root=${HYPER_GSM8K_DATA_ROOT}
        container_data_root=/data/gsm8k
        config_relative=hyper_parallel/rl/examples/gsm8k/configs/codex_multi_turn.yaml
        required_data_files=(train.parquet)
        ;;
    gsm8k_deepseek)
        runtime_image=${HYPER_DEEPSEEK_IMAGE:-${default_image}}
        HYPER_VLLM_MODEL_ROOT=${HYPER_DEEPSEEK_MODEL_ROOT:-${workspace_root}/models/Qwen3-4B}
        task_data_root=${HYPER_DEEPSEEK_DATA_ROOT:-${workspace_root}/data/gsm8k}
        default_deepseek_result=${repo_root}/hyper_parallel/rl/output/qwen3-4b-gsm8k-deepseek
        HYPER_VLLM_RESULT_ROOT=${HYPER_DEEPSEEK_RESULT_ROOT:-${default_deepseek_result}}
        HYPER_VLLM_VISIBLE_DEVICES=${HYPER_DEEPSEEK_VISIBLE_DEVICES:-0,1}
        HYPER_VLLM_MODEL_IMPLEMENTATION=${HYPER_DEEPSEEK_MODEL_IMPLEMENTATION:-native}
        HYPER_VLLM_TIMEOUT_SECONDS=${HYPER_DEEPSEEK_TIMEOUT_SECONDS:-3600}
        HYPER_REQUIRE_LEARNING_UPDATE=${HYPER_DEEPSEEK_REQUIRE_LEARNING_UPDATE:-true}
        HYPER_VLLM_PORT=${HYPER_DEEPSEEK_VLLM_PORT:-8100}
        HYPER_CODEX_GATEWAY_PORT=${HYPER_DEEPSEEK_GATEWAY_PORT:-8300}
        container_data_root=/data/gsm8k
        config_relative=hyper_parallel/rl/examples/gsm8k/configs/deepseek_multi_turn.yaml
        required_data_files=(train.parquet)
        ;;
    *)
        printf 'HYPER_AGENTIC_TASK must be gsm8k, gsm8k_multi, gsm8k_codex, or gsm8k_deepseek, got: %s\n' \
            "${HYPER_AGENTIC_TASK}" >&2
        exit 1
        ;;
esac
hccl_base_port=62600
hccl_socket_range=62600-62700
if [[ "${HYPER_AGENTIC_TASK}" == "gsm8k_deepseek" ]]; then
    hccl_base_port=62800
    hccl_socket_range=62800-62900
    [[ "${HYPER_DEEPSEEK_GPU_MEMORY_UTILIZATION:-0.15}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]] &&
    [[ "${HYPER_DEEPSEEK_KV_CACHE_MEMORY_BYTES:-1073741824}" =~ ^[1-9][0-9]*$ ]] &&
    [[ "${HYPER_DEEPSEEK_MAX_MODEL_LEN:-2048}" =~ ^[1-9][0-9]*$ ]] || {
        printf 'DeepSeek requires GPU utilization in (0, 1] and positive KV bytes/model length\n' >&2
        exit 1
    }
fi
: "${HYPER_VLLM_RESULT_ROOT:=${repo_root}/hyper_parallel/rl/output/qwen3-4b-${HYPER_AGENTIC_TASK}}"

[[ "${HYPER_VLLM_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]] || {
    printf 'HYPER_VLLM_VISIBLE_DEVICES must contain exactly two NPUs\n' >&2
    exit 1
}
IFS=',' read -r first_device second_device <<< "${HYPER_VLLM_VISIBLE_DEVICES}"
[[ "${first_device}" != "${second_device}" ]] || {
    printf 'HYPER_VLLM_VISIBLE_DEVICES must contain two distinct NPUs\n' >&2
    exit 1
}
[[ "${HYPER_VLLM_MODEL_IMPLEMENTATION}" =~ ^(native|hyper)$ ]] || {
    printf 'HYPER_VLLM_MODEL_IMPLEMENTATION must be native or hyper\n' >&2
    exit 1
}
[[ "${HYPER_REQUIRE_LEARNING_UPDATE}" =~ ^(true|false)$ ]] || {
    printf 'HYPER_REQUIRE_LEARNING_UPDATE must be true or false\n' >&2
    exit 1
}
[[ "${HYPER_VLLM_PORT}" =~ ^[0-9]+$ ]] &&
(( HYPER_VLLM_PORT >= 1 && HYPER_VLLM_PORT <= 65535 )) || {
    printf 'HYPER_VLLM_PORT must be an integer between 1 and 65535\n' >&2
    exit 1
}
[[ "${HYPER_CODEX_GATEWAY_PORT}" =~ ^[0-9]+$ ]] &&
(( HYPER_CODEX_GATEWAY_PORT >= 1 && HYPER_CODEX_GATEWAY_PORT <= 65535 )) || {
    printf 'HYPER_CODEX_GATEWAY_PORT must be an integer between 1 and 65535\n' >&2
    exit 1
}
[[ "${HYPER_VLLM_PORT}" != "${HYPER_CODEX_GATEWAY_PORT}" ]] || {
    printf 'HYPER_VLLM_PORT and HYPER_CODEX_GATEWAY_PORT must differ\n' >&2
    exit 1
}
[[ -d "${HYPER_VLLM_MODEL_ROOT}" ]] || {
    printf 'Model directory does not exist: %s\n' "${HYPER_VLLM_MODEL_ROOT}" >&2
    exit 1
}
model_config=${HYPER_VLLM_MODEL_ROOT}/config.json
[[ -f "${model_config}" ]] || {
    printf 'Qwen3-4B config.json does not exist: %s\n' "${model_config}" >&2
    exit 1
}
if [[ "${HYPER_AGENTIC_TASK}" != "gsm8k_deepseek" ]]; then
    grep -Eq '"architectures"[[:space:]]*:' "${model_config}" &&
    grep -Eq '"Qwen3ForCausalLM"' "${model_config}" &&
    grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' "${model_config}" &&
    grep -Eq '"hidden_size"[[:space:]]*:[[:space:]]*2560' "${model_config}" &&
    grep -Eq '"num_hidden_layers"[[:space:]]*:[[:space:]]*36' "${model_config}" || {
        printf 'Model config is not the expected Qwen3-4B identity: %s\n' \
            "${model_config}" >&2
        exit 1
    }
fi
for data_file in "${required_data_files[@]}"; do
    [[ -f "${task_data_root}/${data_file}" ]] || {
        printf 'Required data file does not exist: %s/%s\n' \
            "${task_data_root}" "${data_file}" >&2
        exit 1
    }
done

mkdir -p "${HYPER_VLLM_RESULT_ROOT}"
result_root=$(cd -- "${HYPER_VLLM_RESULT_ROOT}" && pwd)

docker run --rm --privileged --shm-size=64g --network=host \
    -e "ASCEND_RT_VISIBLE_DEVICES=${HYPER_VLLM_VISIBLE_DEVICES}" \
    -e "HYPER_RESULT_UID=$(id -u)" \
    -e "HYPER_RESULT_GID=$(id -g)" \
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e "HCCL_IF_BASE_PORT=${hccl_base_port}" \
    -e "HCCL_NPU_SOCKET_PORT_RANGE=${hccl_socket_range}" \
    -e "HYPER_RUN_TASK=${HYPER_AGENTIC_TASK}" \
    -e "HYPER_RUN_CONFIG=${config_relative}" \
    -e "HYPER_RUN_TIMEOUT_SECONDS=${HYPER_VLLM_TIMEOUT_SECONDS}" \
    -e "HYPER_RUN_MODEL_IMPLEMENTATION=${HYPER_VLLM_MODEL_IMPLEMENTATION}" \
    -e "HYPER_RUN_REQUIRE_LEARNING_UPDATE=${HYPER_REQUIRE_LEARNING_UPDATE}" \
    -e "HYPER_RUN_VLLM_PORT=${HYPER_VLLM_PORT}" \
    -e "HYPER_RUN_GATEWAY_PORT=${HYPER_CODEX_GATEWAY_PORT}" \
    -e "HYPER_RUN_GPU_MEMORY_UTILIZATION=${HYPER_DEEPSEEK_GPU_MEMORY_UTILIZATION:-0.15}" \
    -e "HYPER_RUN_KV_CACHE_MEMORY_BYTES=${HYPER_DEEPSEEK_KV_CACHE_MEMORY_BYTES:-1073741824}" \
    -e "HYPER_RUN_MAX_MODEL_LEN=${HYPER_DEEPSEEK_MAX_MODEL_LEN:-2048}" \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${repo_root}:/workspace/hyper-parallel:ro" \
    -v "${HYPER_VLLM_MODEL_ROOT}:/models/Qwen3-4B:ro" \
    -v "${task_data_root}:${container_data_root}:ro" \
    -v "${result_root}:/results" \
    -w /workspace/hyper-parallel \
    "${runtime_image}" /bin/bash -lc '
        set -euo pipefail
        trap "chown -R ${HYPER_RESULT_UID}:${HYPER_RESULT_GID} /results" EXIT
        if [[ "${HYPER_RUN_TASK}" == *_codex ]]; then
            codex --version | grep -Eq "(^|[[:space:]])0[.]152[.]1($|[[:space:]])"
        elif [[ "${HYPER_RUN_TASK}" == "gsm8k_deepseek" ]]; then
            python -c "from importlib.metadata import version; import deepseek_harness; \
assert version(\"deepseek-harness-sdk\") == \"0.1.1rc1\""
        fi
        unset VLLM_PLUGINS
        export PYTHONPATH=/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel:${PYTHONPATH:-}
        bash /workspace/hyper-parallel/hyper_parallel/rl/docker/install_runtime.sh
        config_path=/workspace/hyper-parallel/${HYPER_RUN_CONFIG}
        log_file=/results/train.log
        args=(
            "${config_path}"
            "--rollout.vllm.model_implementation=${HYPER_RUN_MODEL_IMPLEMENTATION}"
            "--rollout.vllm.port=${HYPER_RUN_VLLM_PORT}"
        )
        if [[ "${HYPER_RUN_TASK}" == "gsm8k" || "${HYPER_RUN_TASK}" == "gsm8k_multi" ]]; then
            args+=(
                --data.answer_column=null
                --data.max_train_samples=4
                --data.shuffle=false
                --evaluation.batch_size=1
                --evaluation.enabled=false
                --evaluation.log_samples=0
                --evaluation.max_new_tokens=256
                --evaluation.max_samples=1
                --evaluation.progress_steps=0
                --logging.backends='[console]'
                --logging.experiment_name=qwen3_4b_gsm8k_single_turn_two_step
                --logging.wandb.directory=/results/wandb
                --logging.wandb.mode=disabled
                --rollout.seed=20260829
                --rollout.vllm.batch_invariant=false
                --rollout.vllm.data_parallel_size=2
                --rollout.vllm.enable_chunked_prefill=false
                --rollout.vllm.enable_prefix_caching=false
                --rollout.vllm.kv_cache_memory_bytes=1073741824
                --rollout.vllm.max_num_batched_tokens=1024
                --rollout.vllm.max_num_seqs=4
                --agentic.max_episode_tokens=1024
                --train.accelerator.dp_shard=2
                --train.checkpoint.output_dir=/results/checkpoints
                --train.checkpoint.save_steps=0
                --train.checkpoint.verify_reload=false
                --train.max_steps=2
                --train.micro_batch_size=1
                --train.prompt_batch_size=1
                --train.response_mini_batch_size=4
            )
        fi
        if [[ "${HYPER_RUN_TASK}" == "gsm8k_deepseek" ]]; then
            args+=(
                "--rollout.vllm.gpu_memory_utilization=${HYPER_RUN_GPU_MEMORY_UTILIZATION}"
                "--rollout.vllm.kv_cache_memory_bytes=${HYPER_RUN_KV_CACHE_MEMORY_BYTES}"
                "--rollout.vllm.max_model_len=${HYPER_RUN_MAX_MODEL_LEN}"
                "--agentic.max_episode_tokens=${HYPER_RUN_MAX_MODEL_LEN}"
                "--agentic.deepseek.gateway_port=${HYPER_RUN_GATEWAY_PORT}"
            )
        fi
        if [[ "${HYPER_RUN_TASK}" == "gsm8k_multi" ]]; then
            args+=(
                --agentic.interaction_mode=multi_turn
                --agentic.apply_chat_template=true
                --agentic.max_turns=4
                --agentic.max_observation_tokens=256
                --logging.experiment_name=qwen3_4b_gsm8k_multi_turn_two_step
            )
        fi
        if [[ "${HYPER_RUN_TASK}" == *_codex ]]; then
            args+=(
                "--agentic.codex.gateway_port=${HYPER_RUN_GATEWAY_PORT}"
            )
        fi
        if [[ "${HYPER_RUN_REQUIRE_LEARNING_UPDATE}" == "true" &&
              ( "${HYPER_RUN_TASK}" == "gsm8k" || "${HYPER_RUN_TASK}" == "gsm8k_multi" ) ]]; then
            args+=(
                --train.learning_gate.enabled=true
                --train.learning_gate.min_gradient_norm=1.0e-12
                --train.learning_gate.require_mixed_rewards=true
            )
        fi
        set +e
        timeout --signal=TERM --kill-after=60s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
            python -m torch.distributed.run --standalone --nproc_per_node=2 \
            hyper_parallel/rl/train_rl.py "${args[@]}" "$@" \
            2>&1 | tee "${log_file}"
        status=${PIPESTATUS[0]}
        set -e
        if (( status != 0 )); then
            exit "${status}"
        fi
        grep -q "step=1 |" "${log_file}"
        grep -q "step=2 |" "${log_file}"
        grep -q "step=2 |.*policy/version=2" "${log_file}"
        grep -q "step=2 |.*train/global_step=2" "${log_file}"
        optimizer_steps_pattern="1([, ]|$)"
        if [[ "${HYPER_RUN_TASK}" == "gsm8k_codex" || "${HYPER_RUN_TASK}" == "gsm8k_deepseek" ]]; then
            optimizer_steps_pattern="[1-9][0-9]*([, ]|$)"
        fi
        grep -Eq "step=2 [|].*train/optimizer_steps=${optimizer_steps_pattern}" "${log_file}"
        checkpoint_manifest=/results/checkpoints/step_2/checkpoint_complete.json
        [[ -f "${checkpoint_manifest}" ]]
        grep -Eq "\"step\"[[:space:]]*:[[:space:]]*2" "${checkpoint_manifest}"
        grep -Eq "\"world_size\"[[:space:]]*:[[:space:]]*2" "${checkpoint_manifest}"
        if [[ "${HYPER_RUN_REQUIRE_LEARNING_UPDATE}" == "true" &&
              "${HYPER_RUN_TASK}" == "gsm8k_deepseek" ]]; then
            gradient_pattern="train/gradient_norm=([1-9][0-9]*(\.[0-9]+)?"
            gradient_pattern+="|0\.[0-9]*[1-9][0-9]*"
            gradient_pattern+="|[1-9][0-9]*(\.[0-9]+)?e-[0-9]+)"
            grep -E "step=[12] \|" "${log_file}" \
                | grep -E "reward/max=1(\.0+)?[, ].*reward/min=0(\.0+)?" \
                | grep -Eq "${gradient_pattern}"
        fi
        if [[ "${HYPER_RUN_REQUIRE_LEARNING_UPDATE}" == "true" &&
              "${HYPER_RUN_TASK}" == *_codex ]]; then
            learning_update_pattern="reward/max=1([.]0+)?([, ]|$).*reward/min=0([.]0+)?([, ]|$).*train/gradient_norm=([1-9][0-9]*([.][0-9]+)?|0[.][0-9]*[1-9][0-9]*)([eE][+-]?[0-9]+)?([, ]|$).*train/optimizer_steps=[1-9][0-9]*([, ]|$)"
            grep -Eq "${learning_update_pattern}" "${log_file}" || {
                printf "Codex learning-update gate failed: no step had mixed rewards, a positive gradient, and an optimizer step. Log: %s\n" \
                    "${log_file}" >&2
                exit 1
            }
            printf "Codex learning-update gate passed. Log: %s\n" "${log_file}"
        fi
        printf "Two-step Qwen3-4B %s Agentic RL control flow passed. Log: %s\n" \
            "${HYPER_RUN_TASK}" "${log_file}"
    ' hyper-rl-trainer "$@"
