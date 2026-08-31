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

: "${HYPER_VLLM_IMAGE:=hyper-parallel/unified-e2-dev:v0.22.1rc1}"
: "${HYPER_VLLM_MODEL_ROOT:=/home/mwl/ckpt/qwen3-4b}"
: "${HYPER_VLLM_DATA_ROOT:=/home/zjy/dataset/hotpotqa}"
: "${HYPER_GSM8K_DATA_ROOT:=/home/zjy/dataset/gsm8k}"
: "${HYPER_AGENTIC_TASK:=search_r1}"
: "${HYPER_VLLM_VISIBLE_DEVICES:=0,1}"
: "${HYPER_VLLM_MODEL_IMPLEMENTATION:=native}"
: "${HYPER_VLLM_TIMEOUT_SECONDS:=3600}"
: "${HYPER_REQUIRE_LEARNING_UPDATE:=false}"

case "${HYPER_AGENTIC_TASK}" in
    gsm8k)
        task_data_root=${HYPER_GSM8K_DATA_ROOT}
        container_data_root=/data/gsm8k
        config_relative=hyper_parallel/rl/examples/agents/gsm8k/configs/single_turn.yaml
        required_data_files=(train.parquet)
        ;;
    search_r1)
        task_data_root=${HYPER_VLLM_DATA_ROOT}
        container_data_root=/data/hotpotqa
        config_relative=hyper_parallel/rl/examples/agents/search_R1/configs/multi_turn.yaml
        required_data_files=(train.parquet corpus.jsonl)
        ;;
    *)
        printf 'HYPER_AGENTIC_TASK must be gsm8k or search_r1, got: %s\n' \
            "${HYPER_AGENTIC_TASK}" >&2
        exit 1
        ;;
esac
: "${HYPER_VLLM_RESULT_ROOT:=$(pwd)/qwen3-4b-${HYPER_AGENTIC_TASK}}"

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
[[ -d "${HYPER_VLLM_MODEL_ROOT}" ]] || {
    printf 'Model directory does not exist: %s\n' "${HYPER_VLLM_MODEL_ROOT}" >&2
    exit 1
}
model_config=${HYPER_VLLM_MODEL_ROOT}/config.json
[[ -f "${model_config}" ]] || {
    printf 'Qwen3-4B config.json does not exist: %s\n' "${model_config}" >&2
    exit 1
}
grep -Eq '"architectures"[[:space:]]*:' "${model_config}" &&
grep -Eq '"Qwen3ForCausalLM"' "${model_config}" &&
grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' "${model_config}" &&
grep -Eq '"hidden_size"[[:space:]]*:[[:space:]]*2560' "${model_config}" &&
grep -Eq '"num_hidden_layers"[[:space:]]*:[[:space:]]*36' "${model_config}" || {
    printf 'Model config is not the expected Qwen3-4B identity: %s\n' \
        "${model_config}" >&2
    exit 1
}
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
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e HCCL_IF_BASE_PORT=62600 \
    -e HCCL_NPU_SOCKET_PORT_RANGE=62600-62700 \
    -e "HYPER_RUN_TASK=${HYPER_AGENTIC_TASK}" \
    -e "HYPER_RUN_CONFIG=${config_relative}" \
    -e "HYPER_RUN_TIMEOUT_SECONDS=${HYPER_VLLM_TIMEOUT_SECONDS}" \
    -e "HYPER_RUN_MODEL_IMPLEMENTATION=${HYPER_VLLM_MODEL_IMPLEMENTATION}" \
    -e "HYPER_RUN_REQUIRE_LEARNING_UPDATE=${HYPER_REQUIRE_LEARNING_UPDATE}" \
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
    "${HYPER_VLLM_IMAGE}" /bin/bash -lc '
        set -euo pipefail
        unset VLLM_PLUGINS
        export PYTHONPATH=/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel:${PYTHONPATH:-}
        config_path=/workspace/hyper-parallel/${HYPER_RUN_CONFIG}
        log_file=/results/train.log
        args=(
            "${config_path}"
            "--rollout.vllm.model_implementation=${HYPER_RUN_MODEL_IMPLEMENTATION}"
        )
        if [[ "${HYPER_RUN_REQUIRE_LEARNING_UPDATE}" == "true" ]]; then
            args+=(
                --train.learning_gate.enabled=true
                --train.learning_gate.min_gradient_norm=1.0e-12
                --train.learning_gate.require_mixed_rewards=true
                --train.learning_gate.require_fingerprint_change=true
            )
        fi
        set +e
        timeout --signal=TERM --kill-after=60s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
            python -m torch.distributed.run --standalone --nproc_per_node=2 \
            hyper_parallel/rl/examples/train_rl.py "${args[@]}" \
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
        grep -q "step=2 |.*train/optimizer_steps=1" "${log_file}"
        checkpoint_manifest=/results/checkpoints/step_2/checkpoint_complete.json
        [[ -f "${checkpoint_manifest}" ]]
        grep -Eq '"step"[[:space:]]*:[[:space:]]*2' "${checkpoint_manifest}"
        grep -Eq '"world_size"[[:space:]]*:[[:space:]]*2' "${checkpoint_manifest}"
        printf "Two-step Qwen3-4B %s Agentic RL control flow passed. Log: %s\n" \
            "${HYPER_RUN_TASK}" "${log_file}"
    '
