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
repo_root=$(cd -- "${script_dir}/../../.." && pwd)

image=${HYPER_DEEPSEEK_IMAGE:-hyper-parallel/hyper-rl-deepseek:v0.22.1rc1}
base_image=${HYPER_RL_IMAGE:-hyper-parallel/hyper-rl:v0.22.1rc1}
proxy=${HYPER_RL_BUILD_PROXY:-http://127.0.0.1:8991}

docker image inspect "${base_image}" >/dev/null 2>&1 || {
    printf 'Base image is unavailable: %s\n' "${base_image}" >&2
    exit 1
}

docker build --network=host \
    --file "${script_dir}/Dockerfile.deepseek" \
    --tag "${image}" \
    --build-arg "BASE_IMAGE=${base_image}" \
    --build-arg "DEEPSEEK_HARNESS_VERSION=0.1.1rc1" \
    --build-arg "http_proxy=${proxy}" \
    --build-arg "https_proxy=${proxy}" \
    --build-arg "HTTP_PROXY=${proxy}" \
    --build-arg "HTTPS_PROXY=${proxy}" \
    "${repo_root}"

docker run --rm "${image}" /bin/bash -lc '
    set -euo pipefail
    python - <<"PY"
from importlib.metadata import version
from importlib.util import find_spec

expected = {
    "deepseek-harness-sdk": "0.1.1rc1",
    "flash-attn-npu": "0.2.0b1",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"Image dependency mismatch: expected={expected}, actual={actual}")
for module in ("deepseek_harness", "flash_attn_npu"):
    if find_spec(module) is None:
        raise RuntimeError(f"Image dependency module is unavailable: {module}")
print(f"DeepSeek image dependencies verified: {actual}")
PY
'
printf 'Built and verified image: %s\n' "${image}"
