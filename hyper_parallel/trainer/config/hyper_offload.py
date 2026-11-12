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
"""Native Trainer activation offload options."""

from dataclasses import dataclass


@dataclass
class HyperOffloadConfig:
    """Control activation residency and host-buffer capacity in MiB."""

    enable: bool = False
    max_resident_activation_mb: int = 32768
    max_offload_activation_mb: int = 65536

    def __post_init__(self) -> None:
        """Validate budgets before training allocates model memory."""
        if not isinstance(self.enable, bool):
            raise TypeError("hyper_offload.enable must be a bool")
        for name in ("max_resident_activation_mb", "max_offload_activation_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"hyper_offload.{name} must be a positive integer")
