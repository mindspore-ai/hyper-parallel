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

# Reuse the common plugin installer and the dependency fixes accepted by the Qwen3 recipe.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rl_root=$(cd -- "$script_dir/../../.." && pwd)
bash "$rl_root/docker/install_runtime.sh"

apply_runtime_patch() {
    local target=$1
    local patch_file=$2
    if patch --batch --fuzz=0 --forward --dry-run -d "$target" -p1 < "$patch_file" >/dev/null; then
        patch --batch --fuzz=0 --forward -d "$target" -p1 < "$patch_file"
    elif ! patch --batch --fuzz=0 --dry-run -R -d "$target" -p1 < "$patch_file" >/dev/null; then
        echo "Runtime patch does not match the installed dependency: $patch_file" >&2
        exit 1
    fi
}
apply_runtime_patch /vllm-workspace/vllm-ascend "$rl_root/docker/patches/ascend-sampler-record-stream.patch"
apply_runtime_patch /vllm-workspace/vllm "$rl_root/docker/patches/vllm-dp-coordinator-timeout.patch"
