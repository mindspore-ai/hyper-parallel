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
"""Common MegaKernel integration API; concrete tools live in sibling packages."""

from hyper_parallel.core.multicore.diagnostics.api import (
    MegaKernelDiagnostics, prepare_mega_kernel_diagnostics,
)
from hyper_parallel.core.multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract
from hyper_parallel.core.multicore.diagnostics.common.schedule import DiagnosticAction, schedule
from hyper_parallel.core.multicore.diagnostics.execution import (
    execution_monitor, inspect_execution, summarize_execution_reports,
)

__all__ = [
    "DiagnosticAction", "MegaKernelDiagnostics", "prepare_mega_kernel_diagnostics",
    "validate_mega_kernel_diagnostic_contract", "schedule",
    "execution_monitor", "inspect_execution", "summarize_execution_reports",
]
