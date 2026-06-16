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
"""Configuration for hyper-offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.hyper_offload.planning.base import ResidencyPlanner


@dataclass
class OffloadConfig:
    """Configuration for the offload system.

    Args:
        max_resident_activation_mb: Maximum resident activation memory on device, in MiB.
        max_offload_activation_mb:  Maximum pinned host memory for offload buffers, in MiB.
            Defaults to 65536 (64 GiB).
        planner: Optional residency planner implementation.

    """

    max_resident_activation_mb: int = 1024
    max_offload_activation_mb: int = 65536
    planner: ResidencyPlanner | None = None
