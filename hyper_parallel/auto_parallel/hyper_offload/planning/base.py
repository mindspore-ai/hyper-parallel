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
"""Planner interfaces."""

from __future__ import annotations

from typing import Protocol

from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import ActivationTrace


class ResidencyPlanner(Protocol):
    """Build a storage residency schedule from a captured activation trace."""

    def build(self, trace: ActivationTrace) -> ResidencySchedule:
        """Build a residency schedule from the given trace."""
