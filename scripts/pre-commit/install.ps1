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
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$ConfigPath = Join-Path $RepoRoot ".pre-commit-config.yaml"

Set-Location $RepoRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python is required but was not found."
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error ".pre-commit-config.yaml was not found at $ConfigPath."
}

try {
    python -m pre_commit --version | Out-Null
} catch {
    Write-Host "Installing pre-commit..."
    python -m pip install --user pre-commit
}

Write-Host "Installing git pre-commit hook..."
python -m pre_commit install --hook-type pre-commit --config "$ConfigPath"

Write-Host "Installing hook environments and dependencies from .pre-commit-config.yaml..."
python -m pre_commit install-hooks --config "$ConfigPath"

Write-Host "pre-commit is ready."
Write-Host "You can verify it with: python -m pre_commit run --all-files"
