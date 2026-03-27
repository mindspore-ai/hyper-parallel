#!/bin/bash
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

CONFIG_PATH="${REPO_ROOT}/.pre-commit-config.yaml"
echo "CONFIG_PATH: ${CONFIG_PATH}"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found."
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo ".pre-commit-config.yaml was not found at ${CONFIG_PATH}."
    exit 1
fi

if ! python3 -m pre_commit --version >/dev/null 2>&1; then
    echo "Installing pre-commit..."
    python3 -m pip install --user pre-commit
fi

echo "Installing git pre-commit hook..."
python3 -m pre_commit install --hook-type pre-commit --config "${CONFIG_PATH}"

echo "Installing hook environments and dependencies from .pre-commit-config.yaml..."
python3 -m pre_commit install-hooks --config "${CONFIG_PATH}"

echo "pre-commit is ready."
echo "You can verify it with: python3 -m pre_commit run --all-files"
