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
"""User-facing switch and optional rank-zero Controller endpoint."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RecoveryConfig:
    """Enable replica recovery with a rank-zero or TaskD-hosted Controller.

    Args:
        enable_worker_reboot: Enable process recovery; disabled by default.
        controller_ip: Rank zero's reachable address; omit to use TaskD.
        controller_port: Rank-zero listening port; defaults to 8000 when an IP is supplied.
    """

    enable_worker_reboot: bool = False
    controller_ip: str = ""
    controller_port: Optional[int] = None

    def __post_init__(self) -> None:
        """Distinguish omitted endpoints from explicitly hosted Controllers."""
        if self.controller_ip and self.controller_port is None:
            object.__setattr__(self, "controller_port", 8000)
        if self.controller_port is not None and not self.controller_ip:
            raise ValueError("controller_port requires controller_ip; omit both to use TaskD")
        if self.controller_port is not None and not 0 < self.controller_port < 65536:
            raise ValueError("controller_port must be between 1 and 65535")
