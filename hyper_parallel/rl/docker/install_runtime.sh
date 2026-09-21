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

# Container startup mounts the checkout read-only. Build the plugin package in
# temporary storage while PYTHONPATH continues to select the mounted source.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rl_source_root=$(cd -- "${script_dir}/.." && pwd)
rl_install_dir=$(mktemp -d /tmp/hyper-rl-install.XXXXXX)
cp "${rl_source_root}/pyproject.toml" "${rl_install_dir}/"
cp -r "${rl_source_root}/rl" "${rl_install_dir}/rl"
python -m pip install --no-deps --no-build-isolation "${rl_install_dir}"
python - <<'PYTHON'
from importlib.metadata import entry_points

expected = "rl.roles.rollout.vllm_plugin:register_hyper_models"
plugins = entry_points(group="vllm.general_plugins")
if not any(plugin.name == "hyper_parallel" and plugin.value == expected for plugin in plugins):
    raise RuntimeError("Hyper-RL vLLM plugin is not installed")
PYTHON
