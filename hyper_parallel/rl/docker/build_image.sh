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
workspace_root=$(cd -- "${repo_root}/.." && pwd)

image=${HYPER_RL_IMAGE:-hyper-parallel/hyper-rl:v0.22.1rc1}
wheel=${HYPER_RL_FA3_WHEEL:-${workspace_root}/reference/flash-attention-npu/dist/flash_attn_npu-0.2.0b1-cp312-cp312-linux_aarch64.whl}
wheel_sha256=9f58e114b77f72079111e2f86fa9750d3be39d1ec9324b309588a540a3e9e12b
proxy=${HYPER_RL_BUILD_PROXY:-http://127.0.0.1:8991}

[[ -f "${wheel}" ]] || {
    printf 'HYPER_RL_FA3_WHEEL is unavailable: %s\n' "${wheel}" >&2
    exit 1
}
read -r actual_sha256 _ < <(sha256sum "${wheel}")
[[ "${actual_sha256}" == "${wheel_sha256}" ]] || {
    printf 'flash-attn-npu wheel SHA256 mismatch: expected=%s actual=%s\n' \
        "${wheel_sha256}" "${actual_sha256}" >&2
    exit 1
}

wheel=$(cd -- "$(dirname -- "${wheel}")" && pwd)/$(basename -- "${wheel}")
wheel_dir=$(dirname -- "${wheel}")
wheel_name=$(basename -- "${wheel}")

printf 'Building %s\n' "${image}"
docker build --network=host \
    --file "${script_dir}/Dockerfile" \
    --tag "${image}" \
    --build-arg "FLASH_ATTN_NPU_WHEEL=${wheel_name}" \
    --build-arg "FLASH_ATTN_NPU_SHA256=${wheel_sha256}" \
    --build-arg "http_proxy=${proxy}" \
    --build-arg "https_proxy=${proxy}" \
    --build-arg "HTTP_PROXY=${proxy}" \
    --build-arg "HTTPS_PROXY=${proxy}" \
    "${wheel_dir}"

docker run --rm "${image}" /bin/bash -lc '
    set -euo pipefail
    python - <<"PY"
from importlib.metadata import version
from importlib.util import find_spec

expected = {
    "batch_invariant_ops": "1.0.0",
    "flash-attn-npu": "0.2.0b1",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"Image dependency mismatch: expected={expected}, actual={actual}")
for module in ("batch_invariant_ops", "flash_attn_npu"):
    if find_spec(module) is None:
        raise RuntimeError(f"Image dependency module is unavailable: {module}")
print(f"Hyper-RL image dependencies verified: {actual}")
PY
'

printf 'Built and verified image: %s\n' "${image}"
