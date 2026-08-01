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

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass
class LowPrecisionConfig:
    """Configure build-time NPU low-precision conversion.

    Phase 1 deliberately exposes only the MXFP8 E4M3 and MX block
    combination. Additional formats become valid only when their kernels and
    end-to-end training paths are implemented.
    """

    enabled: bool = False
    format: Literal["mxfp8_e4m3"] = "mxfp8_e4m3"
    scaling: Literal["mx_block"] = "mx_block"
    include_fqns: list[str] = field(default_factory=list)
    exclude_fqns: list[str] = field(default_factory=list)
    precision_debug: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate supported Phase 1 values and FQN patterns."""

        if not isinstance(self.enabled, bool):
            raise ValueError(
                "LowPrecisionConfig.enabled must be a bool, "
                f"but got {type(self.enabled).__name__}."
            )
        if self.format != "mxfp8_e4m3":
            raise ValueError(
                "LowPrecisionConfig.format must be 'mxfp8_e4m3' in Phase 1, "
                f"but got {self.format!r}."
            )
        if self.scaling != "mx_block":
            raise ValueError(
                "LowPrecisionConfig.scaling must be 'mx_block' in Phase 1, "
                f"but got {self.scaling!r}."
            )
        self._validate_patterns("include_fqns", self.include_fqns)
        self._validate_patterns("exclude_fqns", self.exclude_fqns)
        if self.precision_debug is not None and not isinstance(
            self.precision_debug,
            Mapping,
        ):
            raise ValueError("LowPrecisionConfig.precision_debug must be a mapping")
        if self.precision_debug is not None and not self.enabled:
            raise ValueError(
                "LowPrecisionConfig.precision_debug requires enabled=True"
            )

    @staticmethod
    def _validate_patterns(field_name: str, patterns: list[str]) -> None:
        if not isinstance(patterns, list):
            raise ValueError(
                f"LowPrecisionConfig.{field_name} must be a list of non-empty strings."
            )
        if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise ValueError(
                f"LowPrecisionConfig.{field_name} must contain only non-empty strings."
            )
