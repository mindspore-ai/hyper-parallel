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
"""Public entry points for model-independent logical memory dry-run."""

from importlib import import_module
from typing import Any

__all__ = [
    "DryRunBatchMocker",
    "DryRunConfig",
    "DryRunOpRegistry",
    "DryRunRuntime",
    "DryRunSnapshotConfig",
    "HyperModelsDryRunRunner",
    "LogicalMemoryTracker",
    "ValueDependencyHandler",
    "ValueDependencyStage",
    "get_dryrun_op_registry",
    "install_dryrun_compatibility",
    "register_builtin_dryrun_ops",
    "register_fake_op",
    "register_value_dependency_handler",
    "write_memory_snapshot",
]

_EXPORTS = {
    "DryRunBatchMocker": ("runtime", "DryRunBatchMocker"),
    "DryRunConfig": ("config", "DryRunConfig"),
    "DryRunOpRegistry": ("operator_support", "DryRunOpRegistry"),
    "DryRunRuntime": ("runtime", "DryRunRuntime"),
    "DryRunSnapshotConfig": ("config", "DryRunSnapshotConfig"),
    "HyperModelsDryRunRunner": ("runner", "HyperModelsDryRunRunner"),
    "LogicalMemoryTracker": ("memory", "LogicalMemoryTracker"),
    "ValueDependencyHandler": ("runtime", "ValueDependencyHandler"),
    "ValueDependencyStage": ("runtime", "ValueDependencyStage"),
    "get_dryrun_op_registry": ("operator_support", "get_dryrun_op_registry"),
    "install_dryrun_compatibility": ("compatibility", "install_dryrun_compatibility"),
    "register_builtin_dryrun_ops": ("operator_support", "register_builtin_dryrun_ops"),
    "register_fake_op": ("operator_support", "register_fake_op"),
    "register_value_dependency_handler": ("runtime", "register_value_dependency_handler"),
    "write_memory_snapshot": ("memory", "write_memory_snapshot"),
}


def __getattr__(name: str) -> Any:
    """Load a public dry-run symbol without importing the runner eagerly."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
