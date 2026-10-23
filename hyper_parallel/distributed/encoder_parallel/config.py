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
"""User configuration for the visual encoder parallel runtime."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class EncoderParallelConfig:
    """Configuration of a derived Encoder-DP/Encoder-CP domain."""

    enabled: bool = False
    encoder_cp_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("[HP-ENCODER-002] enabled must be a boolean")
        if (isinstance(self.encoder_cp_size, bool) or not isinstance(self.encoder_cp_size, int)
                or self.encoder_cp_size < 1):
            raise ValueError(
                "[HP-ENCODER-002] accelerator.encoder_parallel.encoder_cp_size "
                f"must be a positive integer, got {self.encoder_cp_size!r}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "EncoderParallelConfig":
        """Parse a mapping while rejecting unknown configuration keys.

        Args:
            raw: User's optional encoder-parallel configuration mapping.
        """
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("[HP-ENCODER-002] encoder_parallel must be a mapping")
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"[HP-ENCODER-002] unknown encoder_parallel keys: {sorted(unknown)}")
        return cls(**raw)
