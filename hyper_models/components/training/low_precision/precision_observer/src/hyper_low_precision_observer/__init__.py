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
"""Framework-neutral low-precision GEMM observation APIs."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from hyper_low_precision_observer.config import (
    DebugOutput,
    DebugOverlay,
    DebugSchedule,
    DebugSection,
    ObserveAction,
    PrecisionDebugProgram,
    Selector,
)
from hyper_low_precision_observer.context import PrecisionContext, metric_key
from hyper_low_precision_observer.metrics import (
    DeviceErrorMetrics,
    ErrorMetrics,
    materialize_device_metrics,
)

if TYPE_CHECKING:
    from hyper_low_precision_observer.report import (
        PrecisionReport,
        aggregate_rank_artifacts,
        generate_offline_reports,
    )

_REPORT_EXPORTS = {
    "PrecisionReport",
    "aggregate_rank_artifacts",
    "generate_offline_reports",
}

__all__ = [
    "DebugOutput",
    "DebugOverlay",
    "DebugSchedule",
    "DebugSection",
    "DeviceErrorMetrics",
    "ErrorMetrics",
    "ObserveAction",
    "PrecisionContext",
    "PrecisionDebugProgram",
    "PrecisionReport",
    "Selector",
    "aggregate_rank_artifacts",
    "generate_offline_reports",
    "materialize_device_metrics",
    "metric_key",
]


def __getattr__(name: str) -> Any:
    """Load offline reporting only when a reporting API is requested."""
    if name not in _REPORT_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("hyper_low_precision_observer.report"), name)
    globals()[name] = value
    return value
