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
"""Typed configuration for NPU low-precision model conversion."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LowPrecisionConfig:
    """Configure build-time NPU low-precision conversion.

    Only format/scaling pairs backed by a complete Dense implementation are
    accepted.
    """

    enabled: bool = False
    format: Literal["mxfp8_e4m3", "hif8"] = "mxfp8_e4m3"
    scaling: Literal["mx_block", "current"] = "mx_block"

    def __post_init__(self) -> None:
        """Validate the supported format/scaling combinations."""

        if not isinstance(self.enabled, bool):
            raise ValueError(
                "LowPrecisionConfig.enabled must be a bool, "
                f"but got {type(self.enabled).__name__}."
            )
        supported = {
            ("mxfp8_e4m3", "mx_block"),
            ("hif8", "current"),
        }
        if (self.format, self.scaling) not in supported:
            raise ValueError(
                "Unsupported low-precision format/scaling combination "
                f"{self.format!r}/{self.scaling!r}; expected one of "
                "mxfp8_e4m3/mx_block or hif8/current."
            )
