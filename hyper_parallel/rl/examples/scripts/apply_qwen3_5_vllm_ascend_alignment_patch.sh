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
patch_file=${repo_root}/hyper_parallel/rl/patches/vllm_ascend/qwen3_5/alignment.patch
vllm_ascend_root=${1:-/vllm-workspace/vllm-ascend}
supported_transformers_version=5.5.4

[[ -d "${vllm_ascend_root}/vllm_ascend" ]] || {
    printf 'vLLM-Ascend source tree is unavailable: %s\n' "${vllm_ascend_root}" >&2
    exit 1
}
command -v python >/dev/null || {
    printf 'Python is required to verify the Transformers alignment dependency\n' >&2
    exit 1
}
transformers_version=$(
    python -c 'from importlib.metadata import version; print(version("transformers").split("+", 1)[0])'
)
[[ "${transformers_version}" == "${supported_transformers_version}" ]] || {
    printf 'Qwen3.5 alignment requires Transformers %s, got %s\n' \
        "${supported_transformers_version}" "${transformers_version}" >&2
    exit 1
}

verify_preimage() {
    local relative_path=$1
    local expected_sha256=$2
    local actual_sha256
    read -r actual_sha256 _ < <(sha256sum "${vllm_ascend_root}/${relative_path}")
    [[ "${actual_sha256}" == "${expected_sha256}" ]] || {
        printf 'Unexpected vLLM-Ascend preimage for %s: expected %s, got %s\n' \
            "${relative_path}" "${expected_sha256}" "${actual_sha256}" >&2
        exit 1
    }
}

verify_preimage vllm_ascend/ops/gdn.py \
    e4fa28c4097c17f1c6c72bfddae5d0fb3aa8accc845963b2b0d585b786a20516
verify_preimage vllm_ascend/patch/worker/patch_qwen3_5.py \
    21fc9a2064f9b490ffcf3d61bacc8ddb67806a4c7490895a33128e65438838d3
verify_preimage vllm_ascend/attention/attention_v1.py \
    b21443d6c4340b1e697a56f6a1f559d15137c2e40888cbf94a4219a73ba01d4b

patch --dry-run --fuzz=0 --forward --batch -d "${vllm_ascend_root}" -p1 < "${patch_file}"
patch --fuzz=0 --forward --batch -d "${vllm_ascend_root}" -p1 < "${patch_file}"
