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
"""Preflight integrity checks over codegen artifact bundles.

Per-zone verifiers composed by ``preflight_integrity_check`` (manager) and
callable individually by the CLI.
"""
from hyper_parallel.codegen.check.drift import verify_snapshot_drift
from hyper_parallel.codegen.check.preflight import (
    verify_boundary_forms,
    verify_generated_source,
    verify_meta_required_fields,
    verify_output_hashes,
    verify_param_plan,
    verify_signature,
    verify_snapshot_contract,
    warn_skipped_overrides,
)

__all__ = [
    "verify_boundary_forms",
    "verify_generated_source",
    "verify_meta_required_fields",
    "verify_output_hashes",
    "verify_param_plan",
    "verify_signature",
    "verify_snapshot_contract",
    "verify_snapshot_drift",
    "warn_skipped_overrides",
]
