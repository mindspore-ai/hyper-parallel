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

image=${HYPER_QWEN3_IMAGE:-hyper-parallel/unified-e2-dev:v0.22.1rc1}
model_root=${HYPER_QWEN3_MODEL_ROOT:-${workspace_root}/models/Qwen3-4B}
data_root=${HYPER_QWEN3_DATA_ROOT:-${workspace_root}/data/gsm8k}
result_root=${HYPER_QWEN3_RESULT_ROOT:-${repo_root}/.rollout-results/qwen3-consistency-smoke}
visible_devices=${HYPER_QWEN3_VISIBLE_DEVICES:-0,1}
wheel=${HYPER_QWEN3_FA3_WHEEL:-}
wheel_sha256=${HYPER_QWEN3_FA3_WHEEL_SHA256:-9f58e114b77f72079111e2f86fa9750d3be39d1ec9324b309588a540a3e9e12b}
timeout_seconds=${HYPER_QWEN3_TIMEOUT_SECONDS:-3600}
hccl_if_base_port=${HCCL_IF_BASE_PORT:-62200}
hccl_socket_port_range=${HCCL_NPU_SOCKET_PORT_RANGE:-62200-62300}

[[ "${visible_devices}" =~ ^[0-9]+,[0-9]+$ ]] || {
    printf 'HYPER_QWEN3_VISIBLE_DEVICES must contain exactly two NPUs, got: %s\n' \
        "${visible_devices}" >&2
    exit 1
}
first_device=${visible_devices%%,*}
second_device=${visible_devices#*,}
((10#${first_device} != 10#${second_device})) || {
    printf 'HYPER_QWEN3_VISIBLE_DEVICES must contain two distinct NPUs, got: %s\n' \
        "${visible_devices}" >&2
    exit 1
}
[[ -d "${model_root}" ]] || {
    printf 'Qwen3 model directory is unavailable: %s\n' "${model_root}" >&2
    exit 1
}
[[ -f "${data_root}/train.parquet" && -f "${data_root}/test.parquet" ]] || {
    printf 'GSM8K train.parquet and test.parquet are required under: %s\n' "${data_root}" >&2
    exit 1
}
[[ -n "${wheel}" && -f "${wheel}" ]] || {
    printf 'HYPER_QWEN3_FA3_WHEEL must point to the flash-attn-npu 0.2.0b1 wheel\n' >&2
    exit 1
}
read -r actual_wheel_sha256 _ < <(sha256sum "${wheel}")
[[ "${actual_wheel_sha256}" == "${wheel_sha256}" ]] || {
    printf 'flash-attn-npu wheel SHA256 mismatch: expected=%s actual=%s\n' \
        "${wheel_sha256}" "${actual_wheel_sha256}" >&2
    exit 1
}

mkdir -p "${result_root}"
wheel_name=$(basename "${wheel}")

docker run --rm --privileged --shm-size=12g --network=host \
    --entrypoint /bin/bash \
    -e "ASCEND_RT_VISIBLE_DEVICES=${visible_devices}" \
    -e HYPER_PARALLEL_PLATFORM=torch \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo \
    -e "HCCL_IF_BASE_PORT=${hccl_if_base_port}" \
    -e "HCCL_NPU_SOCKET_PORT_RANGE=${hccl_socket_port_range}" \
    -e "HYPER_RUN_TIMEOUT_SECONDS=${timeout_seconds}" \
    -e "HYPER_WHEEL_NAME=${wheel_name}" \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${repo_root}:/workspace/hyper-parallel:ro" \
    -v "${model_root}:/models/Qwen3-4B:ro" \
    -v "${data_root}:/data/gsm8k:ro" \
    -v "${result_root}:/results" \
    -v "${wheel}:/wheels/${wheel_name}:ro" \
    -w /workspace/hyper-parallel \
    "${image}" -lc '
        set -euo pipefail
        unset VLLM_PLUGINS
        source_dir=$(mktemp -d /tmp/hyper-rl-src.XXXXXX)
        site_dir=$(mktemp -d /tmp/hyper-rl-site.XXXXXX)
        cp -a /workspace/hyper-parallel/. "${source_dir}/"
        rm -rf "${source_dir}/.git" "${source_dir}/.rollout-results" \
            "${source_dir}/hyper_parallel/.commit_id"

        python -m pip install --no-deps "/wheels/${HYPER_WHEEL_NAME}" >/dev/null
        python -m pip install --no-build-isolation --no-deps --ignore-installed \
            --target "${site_dir}" -e "${source_dir}" >/dev/null
        export PYTHONPATH="${site_dir}:${source_dir}/hyper_parallel/rl:${source_dir}:${PYTHONPATH:-}"

        selection_dir=/results/gsm8k-qwen3-m3.next
        rm -rf "${selection_dir}"
        timeout --signal=TERM --kill-after=30s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
            python "${source_dir}/hyper_parallel/rl/examples/scripts/select_gsm8k_m3.py" \
            --source /data/gsm8k/train.parquet \
            --model /models/Qwen3-4B \
            --output-dir "${selection_dir}" \
            --implementation native \
            --architecture Qwen3ForCausalLM \
            --candidate-offset 79 \
            --candidate-limit 80 \
            --sample-count 1 \
            --response-count 4 \
            --output-repeats 2 \
            --max-tokens 256 \
            --max-model-len 512
        test -f "${selection_dir}/train.parquet"
        test -f "${selection_dir}/manifest.json"
        rm -rf /results/gsm8k-qwen3-m3
        mv "${selection_dir}" /results/gsm8k-qwen3-m3

        set +e
        timeout --signal=TERM --kill-after=30s "${HYPER_RUN_TIMEOUT_SECONDS}s" \
            python -m torch.distributed.run --standalone --nproc_per_node=2 \
            "${source_dir}/hyper_parallel/rl/examples/train_rl.py" \
            "${source_dir}/hyper_parallel/rl/examples/configs/local_qwen3_4b_gsm8k_vllm_m3.yaml" \
            --consistency.profile=qwen3_ascend_fa3_batch_invariant_v1 \
            --train.max_steps=1 \
            2>&1 | tee /results/qwen3-consistency-smoke.log
        status="${PIPESTATUS[0]}"
        set -e
        exit "${status}"
    '
