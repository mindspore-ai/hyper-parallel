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
"""Shared intermediate models for tracing, planning, and execution."""

from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyAction,
    ResidencyActionType,
    ResidencySchedule,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
    StorageAccess,
    TraceOp,
)

__all__ = [
    "AccessKind",
    "ActivationTrace",
    "ResidencyAction",
    "ResidencyActionType",
    "ResidencySchedule",
    "StorageAccess",
    "OpGuide",
    "TraceOp",
]
