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

action=${1:-}
image=${HYPER_VLLM_IMAGE:-hyper-parallel/unified-e2-dev:v0.22.1rc1}
model_root=${HYPER_VLLM_MODEL_ROOT:-${workspace_root}/models/Qwen3.5-0.8B-Base}
data_root=${HYPER_VLLM_DATA_ROOT:-${workspace_root}/data/gsm8k}
result_root=${HYPER_VLLM_RESULT_ROOT:-${repo_root}/.rollout-results}
network_mode=${HYPER_VLLM_NETWORK_MODE:-bridge}
timeout_seconds=${HYPER_VLLM_TIMEOUT_SECONDS:-600}
model_implementation=${HYPER_VLLM_MODEL_IMPLEMENTATION:-hyper}
detached=${HYPER_VLLM_DETACHED:-false}
container_name=${HYPER_VLLM_CONTAINER_NAME:-hyper-vllm-${model_implementation}-${action}-$$}
log_suffix=${HYPER_VLLM_LOG_SUFFIX:-}
hccl_if_base_port=${HCCL_IF_BASE_PORT:-}
hccl_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE:-}
production_canary_marker=${result_root}/.production-canary-${model_implementation}
production_profile_identity=

usage() {
    printf 'Usage: %s {rollout-tp1|rollout-tp2|rollout-compare-tp1|refit|production-benchmark|grpo-smoke|grpo-colocated-dp-smoke|grpo-m3-select|grpo-m3-2step|grpo-m3-soak|grpo-production-canary|grpo-production}\n' "$0"
}

require_directory() {
    [[ -d "$1" ]] || {
        printf 'Required directory is unavailable: %s\n' "$1" >&2
        exit 1
    }
}

compute_production_profile_identity() {
    local image_id
    local file_hashes
    local identity
    image_id=$(docker image inspect --format '{{.Id}}' "${image}")
    file_hashes=$(
        while IFS= read -r -d '' file; do
            sha256sum "${repo_root}/${file}"
        done < <(git -C "${repo_root}" ls-files --cached --others --exclude-standard -z)
        sha256sum \
            "${model_root}/config.json" \
            "${model_root}/model.safetensors-00001-of-00001.safetensors" \
            "${model_root}/model.safetensors.index.json" \
            "${model_root}/tokenizer.json" \
            "${data_root}/train.parquet" \
            "${data_root}/test.parquet"
    )
    read -r identity _ < <(
        printf '%s\n' "${image_id}" "${model_implementation}" "${file_hashes}" | sha256sum
    )
    printf '%s\n' "${identity}"
}

[[ -n "${action}" ]] || {
    usage
    exit 1
}
[[ "${model_implementation}" == "hyper" || "${model_implementation}" == "native" ]] || {
    printf 'HYPER_VLLM_MODEL_IMPLEMENTATION must be hyper or native, got: %s\n' "${model_implementation}" >&2
    exit 1
}
[[ "${detached}" == "true" || "${detached}" == "false" ]] || {
    printf 'HYPER_VLLM_DETACHED must be true or false, got: %s\n' "${detached}" >&2
    exit 1
}
[[ "${log_suffix}" =~ ^[A-Za-z0-9._-]*$ ]] || {
    printf 'HYPER_VLLM_LOG_SUFFIX contains unsupported characters: %s\n' "${log_suffix}" >&2
    exit 1
}
require_directory "${model_root}"
require_directory "${data_root}"

visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0}
tensor_parallel_size=1
test_script=tests/torch/rl/vllm/validate_qwen3_5_rollout.py
output_file=/results/${model_implementation}-tp1.json
nproc_per_node=1
grpo_config=hyper_parallel/rl/examples/configs/local_qwen3_5_0_8b_gsm8k_vllm_smoke.yaml
case "${action}" in
    rollout-tp1) ;;
    rollout-tp2)
        visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
        tensor_parallel_size=2
        output_file=/results/${model_implementation}-tp2.json
        ;;
    rollout-compare-tp1)
        test_script=tests/torch/rl/vllm/compare_qwen3_5_rollout_reports.py
        output_file=/results/native-hyper-tp1-comparison.json
        ;;
    refit)
        test_script=tests/torch/rl/vllm/validate_qwen3_5_refit.py
        output_file=/results/${model_implementation}-refit-tp1.json
        ;;
    production-benchmark)
        test_script=tests/torch/rl/vllm/benchmark_qwen3_5_production.py
        output_file=/results/${model_implementation}-production-benchmark.json
        if [[ -z "${HYPER_VLLM_TIMEOUT_SECONDS:-}" ]]; then
            timeout_seconds=1800
        fi
        ;;
    grpo-smoke)
        visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
        test_script=hyper_parallel/rl/examples/train_rl.py
        ;;
    grpo-colocated-dp-smoke)
        visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
        test_script=hyper_parallel/rl/examples/train_rl.py
        nproc_per_node=2
        grpo_config=hyper_parallel/rl/examples/configs/local_qwen3_5_0_8b_gsm8k_vllm_colocated_dp_smoke.yaml
        ;;
    grpo-m3-select)
        test_script=hyper_parallel/rl/examples/scripts/select_gsm8k_m3.py
        ;;
    grpo-m3-2step|grpo-m3-soak)
        visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
        test_script=hyper_parallel/rl/examples/train_rl.py
        nproc_per_node=2
        grpo_config=hyper_parallel/rl/examples/configs/local_qwen3_5_0_8b_gsm8k_vllm_m3.yaml
        ;;
    grpo-production-canary|grpo-production)
        [[ -n "${HYPER_VLLM_RESULT_ROOT:-}" ]] || {
            printf 'grpo-production requires a unique HYPER_VLLM_RESULT_ROOT\n' >&2
            exit 1
        }
        [[ -n "${hccl_if_base_port}" && -n "${hccl_socket_port_range}" ]] || {
            printf 'grpo-production requires HCCL_IF_BASE_PORT and HCCL_NPU_SOCKET_PORT_RANGE\n' >&2
            exit 1
        }
        production_profile_identity=$(compute_production_profile_identity)
        if [[ "${action}" == "grpo-production-canary" ]]; then
            rm -f "${production_canary_marker}"
        else
            [[ -f "${production_canary_marker}" ]] || {
                printf 'grpo-production requires a successful current-profile canary: %s\n' \
                    "${production_canary_marker}" >&2
                exit 1
            }
            canary_identity=$(<"${production_canary_marker}")
            [[ "${canary_identity}" == "${production_profile_identity}" ]] || {
                printf 'grpo-production canary is stale; rerun grpo-production-canary\n' >&2
                exit 1
            }
        fi
        visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
        test_script=hyper_parallel/rl/examples/train_rl.py
        nproc_per_node=2
        grpo_config=hyper_parallel/rl/examples/configs/qwen3_5_0_8b_gsm8k_vllm_production.yaml
        if [[ "${action}" == "grpo-production-canary" && -z "${HYPER_VLLM_TIMEOUT_SECONDS:-}" ]]; then
            timeout_seconds=1800
        elif [[ -z "${HYPER_VLLM_TIMEOUT_SECONDS:-}" ]]; then
            timeout_seconds=259200
        fi
        ;;
    *)
        usage
        exit 1
        ;;
esac
mkdir -p "${result_root}"

common_test_args=(
    --model /models/Qwen3.5-0.8B-Base
    --max-tokens 4
    --max-model-len 512
    --gpu-memory-utilization 0.7
    --output "${output_file}"
)
if [[ "${action}" == "grpo-m3-select" ]]; then
    common_test_args=(
        --source /data/gsm8k/train.parquet
        --model /models/Qwen3.5-0.8B-Base
        --output-dir /results/gsm8k-m3
        --implementation "${model_implementation}"
    )
elif [[ "${action}" == "rollout-compare-tp1" ]]; then
    common_test_args=(
        --native /results/native-tp1.json
        --hyper /results/hyper-tp1.json
        --output "${output_file}"
    )
elif [[ "${action}" == "production-benchmark" ]]; then
    common_test_args=(
        --source /data/gsm8k/train.parquet
        --model /models/Qwen3.5-0.8B-Base
        --output "${output_file}"
        --implementation "${model_implementation}"
        --visible-devices "${visible_devices}"
        --batch-invariant
    )
elif [[ "${action}" == "grpo-smoke" || "${action}" == "grpo-colocated-dp-smoke" \
    || "${action}" == "grpo-m3-2step" || "${action}" == "grpo-m3-soak" \
    || "${action}" == "grpo-production-canary" || "${action}" == "grpo-production" ]]; then
    common_test_args=(
        "${grpo_config}"
        --rollout.vllm.model_implementation="${model_implementation}"
    )
    if [[ "${action}" == "grpo-smoke" ]]; then
        rollout_visible_device=${visible_devices#*,}
        rollout_visible_device=${rollout_visible_device%%,*}
        [[ "${rollout_visible_device}" != "${visible_devices}" && -n "${rollout_visible_device}" ]] || {
            printf 'grpo-smoke requires at least two ASCEND_RT_VISIBLE_DEVICES\n' >&2
            exit 1
        }
        common_test_args+=(
            --rollout.vllm.visible_devices="${rollout_visible_device}"
        )
    elif [[ "${action}" == "grpo-m3-soak" ]]; then
        common_test_args+=(
            --train.max_steps=20
            --train.learning_gate.enabled=false
        )
    elif [[ "${action}" == "grpo-production-canary" ]]; then
        common_test_args+=(
            --train.max_steps=1
            --evaluation.enabled=false
            --train.checkpoint.save_steps=0
            --train.checkpoint.save_final=false
            --train.checkpoint.verify_reload=false
            --logging.log_steps=1
        )
    fi
elif [[ "${action}" == "refit" ]]; then
    common_test_args+=(
        --implementation "${model_implementation}"
    )
else
    common_test_args+=(
        --implementation "${model_implementation}"
        --tensor-parallel-size "${tensor_parallel_size}"
    )
fi

docker_lifecycle_args=(--rm)
if [[ "${detached}" == "true" ]]; then
    docker_lifecycle_args=(--detach --name "${container_name}")
fi
docker_hccl_args=()
if [[ -n "${hccl_if_base_port}" ]]; then
    docker_hccl_args+=(-e "HCCL_IF_BASE_PORT=${hccl_if_base_port}")
fi
if [[ -n "${hccl_socket_port_range}" ]]; then
    docker_hccl_args+=(-e "HCCL_NPU_SOCKET_PORT_RANGE=${hccl_socket_port_range}")
fi

docker run "${docker_lifecycle_args[@]}" --privileged --shm-size=12g --network="${network_mode}" \
    "${docker_hccl_args[@]}" \
    -e "ASCEND_RT_VISIBLE_DEVICES=${visible_devices}" \
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e HYPER_VLLM_NUMERICAL_PROFILE=functional \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e "HYPER_RUN_ACTION=${action}" \
    -e "HYPER_RUN_IMPLEMENTATION=${model_implementation}" \
    -e "HYPER_RUN_LOG_SUFFIX=${log_suffix}" \
    -e "HYPER_RUN_TIMEOUT_SECONDS=${timeout_seconds}" \
    -e "HYPER_PRODUCTION_PROFILE_IDENTITY=${production_profile_identity}" \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${repo_root}:/workspace/hyper-parallel:ro" \
    -v "${model_root}:/models/Qwen3.5-0.8B-Base:ro" \
    -v "${data_root}:/data/gsm8k:ro" \
    -v "${result_root}:/results" \
    -w /workspace/hyper-parallel \
    "${image}" /bin/bash -lc '
        set -o pipefail
        unset VLLM_PLUGINS
        source_dir=$(mktemp -d /tmp/hyper-rl-src.XXXXXX)
        site_dir=$(mktemp -d /tmp/hyper-rl-site.XXXXXX)
        cp -a /workspace/hyper-parallel/. "${source_dir}"/
        rm -rf "${source_dir}/.git" "${source_dir}/.rollout-results" \
            "${source_dir}/hyper_parallel/.commit_id"
        python -m pip install --no-build-isolation --no-deps --ignore-installed \
            --target "${site_dir}" -e "${source_dir}" >/dev/null
        export PYTHONPATH="${site_dir}:${source_dir}/hyper_parallel/rl:${source_dir}:/vllm-workspace/vllm:${PYTHONPATH:-}"
        log_file="/results/${HYPER_RUN_IMPLEMENTATION}-${HYPER_RUN_ACTION}${HYPER_RUN_LOG_SUFFIX}.log"
        if [[ "'"${action}"'" == "grpo-smoke" || "'"${action}"'" == "grpo-colocated-dp-smoke" \
            || "'"${action}"'" == "grpo-m3-2step" || "'"${action}"'" == "grpo-m3-soak" \
            || "'"${action}"'" == "grpo-production-canary" || "'"${action}"'" == "grpo-production" ]]; then
            timeout --signal=TERM --kill-after=30s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
                python -m torch.distributed.run --standalone --nproc_per_node='"${nproc_per_node}"' \
                "${source_dir}/'"${test_script}"'" '"${common_test_args[*]}"' 2>&1 | tee "${log_file}"
            status="${PIPESTATUS[0]}"
            if [[ "${HYPER_RUN_ACTION}" == "grpo-production-canary" && "${status}" -eq 0 ]]; then
                printf '%s\n' "${HYPER_PRODUCTION_PROFILE_IDENTITY}" \
                    > "/results/.production-canary-${HYPER_RUN_IMPLEMENTATION}"
            fi
            exit "${status}"
        fi
        timeout --signal=TERM --kill-after=30s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
            python "${source_dir}/'"${test_script}"'" '"${common_test_args[*]}"' 2>&1 | tee "${log_file}"
        exit "${PIPESTATUS[0]}"
    '
