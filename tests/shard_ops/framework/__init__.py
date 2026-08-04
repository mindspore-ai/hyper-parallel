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
"""Public API for the shard ops test framework."""
from tests.shard_ops.framework.case_spec import (
    CompareSpec,
    DerivedSpec,
    InputSpec,
    OpShardCase,
    OpSpec,
    CaseSpec,
    PlacementSpec,
)
from tests.shard_ops.framework.registry import (
    register,
    register_op_family,
    load_cases_from_package,
    load_case_plan_from_package,
)
from tests.shard_ops.framework.suite import GroupSpec, build_suite_groups
from tests.shard_ops.framework.runner import RUNNER

__all__ = [
    "CompareSpec",
    "DerivedSpec",
    "InputSpec",
    "OpShardCase",
    "OpSpec",
    "CaseSpec",
    "PlacementSpec",
    "register",
    "register_op_family",
    "load_cases_from_package",
    "load_case_plan_from_package",
    "GroupSpec",
    "build_suite_groups",
    "RUNNER",
]
