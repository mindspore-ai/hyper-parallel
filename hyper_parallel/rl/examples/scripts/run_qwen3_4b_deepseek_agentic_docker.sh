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

: "${HYPER_DEEPSEEK_IMAGE:=hyper-parallel/hyper-rl-deepseek:v0.22.1rc1}"
: "${HYPER_DEEPSEEK_MODEL_ROOT:=/home/mwl/ckpt/qwen3-4b}"
: "${HYPER_DEEPSEEK_DATA_ROOT:=/home/zjy/dataset/gsm8k}"
: "${HYPER_DEEPSEEK_RESULT_ROOT:=$(pwd)/qwen3-4b-gsm8k-deepseek}"
: "${HYPER_DEEPSEEK_VISIBLE_DEVICES:=0,1}"
: "${HYPER_DEEPSEEK_MODEL_IMPLEMENTATION:=native}"
: "${HYPER_DEEPSEEK_TIMEOUT_SECONDS:=3600}"
: "${HYPER_DEEPSEEK_REQUIRE_LEARNING_UPDATE:=true}"
: "${HYPER_DEEPSEEK_VLLM_PORT:=8100}"
: "${HYPER_DEEPSEEK_GATEWAY_PORT:=8300}"
: "${HYPER_DEEPSEEK_GPU_MEMORY_UTILIZATION:=0.15}"
: "${HYPER_DEEPSEEK_KV_CACHE_MEMORY_BYTES:=1073741824}"
: "${HYPER_DEEPSEEK_MAX_MODEL_LEN:=2048}"

[[ "${HYPER_DEEPSEEK_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]] || {
    printf 'HYPER_DEEPSEEK_VISIBLE_DEVICES must contain exactly two NPUs\n' >&2
    exit 1
}
IFS=',' read -r first_device second_device <<< "${HYPER_DEEPSEEK_VISIBLE_DEVICES}"
[[ "${first_device}" != "${second_device}" ]] || {
    printf 'HYPER_DEEPSEEK_VISIBLE_DEVICES must contain distinct NPUs\n' >&2
    exit 1
}
[[ "${HYPER_DEEPSEEK_MODEL_IMPLEMENTATION}" =~ ^(native|hyper)$ ]] || {
    printf 'HYPER_DEEPSEEK_MODEL_IMPLEMENTATION must be native or hyper\n' >&2
    exit 1
}
[[ "${HYPER_DEEPSEEK_REQUIRE_LEARNING_UPDATE}" =~ ^(true|false)$ ]] || {
    printf 'HYPER_DEEPSEEK_REQUIRE_LEARNING_UPDATE must be true or false\n' >&2
    exit 1
}
[[ "${HYPER_DEEPSEEK_GPU_MEMORY_UTILIZATION}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]] || {
    printf 'HYPER_DEEPSEEK_GPU_MEMORY_UTILIZATION must be in (0, 1]\n' >&2
    exit 1
}
[[ "${HYPER_DEEPSEEK_KV_CACHE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_DEEPSEEK_KV_CACHE_MEMORY_BYTES must be a positive integer\n' >&2
    exit 1
}
[[ "${HYPER_DEEPSEEK_MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'HYPER_DEEPSEEK_MAX_MODEL_LEN must be a positive integer\n' >&2
    exit 1
}
for port in "${HYPER_DEEPSEEK_VLLM_PORT}" "${HYPER_DEEPSEEK_GATEWAY_PORT}"; do
    [[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || {
        printf 'DeepSeek ports must be integers between 1 and 65535\n' >&2
        exit 1
    }
done
[[ "${HYPER_DEEPSEEK_VLLM_PORT}" != "${HYPER_DEEPSEEK_GATEWAY_PORT}" ]] || {
    printf 'vLLM and DeepSeek gateway ports must differ\n' >&2
    exit 1
}
[[ -f "${HYPER_DEEPSEEK_MODEL_ROOT}/config.json" ]] || {
    printf 'Model config does not exist: %s/config.json\n' "${HYPER_DEEPSEEK_MODEL_ROOT}" >&2
    exit 1
}
[[ -f "${HYPER_DEEPSEEK_DATA_ROOT}/train.parquet" ]] || {
    printf 'GSM8K train.parquet does not exist: %s\n' "${HYPER_DEEPSEEK_DATA_ROOT}" >&2
    exit 1
}
mkdir -p "${HYPER_DEEPSEEK_RESULT_ROOT}"
result_root=$(cd -- "${HYPER_DEEPSEEK_RESULT_ROOT}" && pwd)

docker run --rm --privileged --shm-size=64g --network=host \
    -e "ASCEND_RT_VISIBLE_DEVICES=${HYPER_DEEPSEEK_VISIBLE_DEVICES}" \
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e HCCL_IF_BASE_PORT=62800 \
    -e HCCL_NPU_SOCKET_PORT_RANGE=62800-62900 \
    -e "HYPER_RUN_TIMEOUT_SECONDS=${HYPER_DEEPSEEK_TIMEOUT_SECONDS}" \
    -e "HYPER_RUN_MODEL_IMPLEMENTATION=${HYPER_DEEPSEEK_MODEL_IMPLEMENTATION}" \
    -e "HYPER_RUN_REQUIRE_LEARNING_UPDATE=${HYPER_DEEPSEEK_REQUIRE_LEARNING_UPDATE}" \
    -e "HYPER_RUN_VLLM_PORT=${HYPER_DEEPSEEK_VLLM_PORT}" \
    -e "HYPER_RUN_DEEPSEEK_GATEWAY_PORT=${HYPER_DEEPSEEK_GATEWAY_PORT}" \
    -e "HYPER_RUN_GPU_MEMORY_UTILIZATION=${HYPER_DEEPSEEK_GPU_MEMORY_UTILIZATION}" \
    -e "HYPER_RUN_KV_CACHE_MEMORY_BYTES=${HYPER_DEEPSEEK_KV_CACHE_MEMORY_BYTES}" \
    -e "HYPER_RUN_MAX_MODEL_LEN=${HYPER_DEEPSEEK_MAX_MODEL_LEN}" \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${repo_root}:/workspace/hyper-parallel:ro" \
    -v "${HYPER_DEEPSEEK_MODEL_ROOT}:/models/Qwen3-4B:ro" \
    -v "${HYPER_DEEPSEEK_DATA_ROOT}:/data/gsm8k:ro" \
    -v "${result_root}:/results" \
    -w /workspace/hyper-parallel \
    "${HYPER_DEEPSEEK_IMAGE}" /bin/bash -lc '
        set -euo pipefail
        python -c \
            "from importlib.metadata import version; import deepseek_harness; \
assert version(\"deepseek-harness-sdk\") == \"0.1.1rc1\""
        unset VLLM_PLUGINS
        export PYTHONPATH=/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel:${PYTHONPATH:-}
        config_path=/workspace/hyper-parallel/hyper_parallel/rl/examples/agents/gsm8k/configs/deepseek_multi_turn.yaml
        args=(
            "${config_path}"
            "--rollout.vllm.model_implementation=${HYPER_RUN_MODEL_IMPLEMENTATION}"
            "--rollout.vllm.port=${HYPER_RUN_VLLM_PORT}"
            "--rollout.vllm.gpu_memory_utilization=${HYPER_RUN_GPU_MEMORY_UTILIZATION}"
            "--rollout.vllm.kv_cache_memory_bytes=${HYPER_RUN_KV_CACHE_MEMORY_BYTES}"
            "--rollout.vllm.max_model_len=${HYPER_RUN_MAX_MODEL_LEN}"
            "--agentic.max_episode_tokens=${HYPER_RUN_MAX_MODEL_LEN}"
            "--agentic.deepseek.gateway_port=${HYPER_RUN_DEEPSEEK_GATEWAY_PORT}"
        )
        log_file=/results/train.log
        set +e
        timeout --signal=TERM --kill-after=60s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
            python -m torch.distributed.run --standalone --nproc_per_node=2 \
            hyper_parallel/rl/examples/train_rl.py "${args[@]}" 2>&1 | tee "${log_file}"
        status=${PIPESTATUS[0]}
        set -e
        (( status == 0 )) || exit "${status}"
        grep -q "step=1 |" "${log_file}"
        grep -q "step=2 |.*policy/version=2" "${log_file}"
        grep -q "step=2 |.*train/global_step=2" "${log_file}"
        grep -q "step=2 |.*train/optimizer_steps=1" "${log_file}"
        if [[ "${HYPER_RUN_REQUIRE_LEARNING_UPDATE}" == "true" ]]; then
            gradient_pattern="train/gradient_norm=([1-9][0-9]*(\.[0-9]+)?"
            gradient_pattern+="|0\.[0-9]*[1-9][0-9]*"
            gradient_pattern+="|[1-9][0-9]*(\.[0-9]+)?e-[0-9]+)"
            grep -E "step=[12] \|.*policy/fingerprint_changed=1" "${log_file}" \
                | grep -E "reward/max=1(\.0+)?[, ].*reward/min=0(\.0+)?" \
                | grep -Eq "${gradient_pattern}"
        fi
        manifest=/results/checkpoints/step_2/checkpoint_complete.json
        [[ -f "${manifest}" ]]
        grep -Eq "\"step\"[[:space:]]*:[[:space:]]*2" "${manifest}"
        grep -Eq "\"world_size\"[[:space:]]*:[[:space:]]*2" "${manifest}"
        printf "Two-step Qwen3-4B DeepSeek Harness Agentic RL passed. Log: %s\n" "${log_file}"
    '
