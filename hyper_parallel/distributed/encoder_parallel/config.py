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
    """Configuration of a derived Encoder-DP/Encoder-CP domain.

    ``packed_attention`` exchanges the concatenated bucket sequence and calls
    image-isolated TND attention. ``projector_mode`` selects leader projection
    after a full gather, or spatially sharded projection with direct input dispatch.
    Defaults retain per-image attention, leader projection, and patch-square
    work assignment. ``overlap`` splits Q/K/V projection and overlaps input
    communication in forward and backward; it requires packed attention.
    ``balance_cost`` optionally includes the model's linear projection/MLP work.
    """

    enabled: bool = False
    encoder_cp_size: int = 1
    packed_attention: bool = False
    projector_mode: str = "leader"
    overlap: bool = False
    balance_cost: str = "attention"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.packed_attention, bool):
            raise ValueError("packed_attention must be a boolean")
        if not isinstance(self.overlap, bool):
            raise ValueError("overlap must be a boolean")
        if self.overlap and not self.packed_attention:
            raise ValueError("encoder_parallel.overlap requires packed_attention=True")
        if self.projector_mode not in ("leader", "sharded_dispatch"):
            raise ValueError("projector_mode must be leader or sharded_dispatch")
        if self.balance_cost not in ("attention", "encoder_flops"):
            raise ValueError("balance_cost must be attention or encoder_flops")
        if (isinstance(self.encoder_cp_size, bool) or not isinstance(self.encoder_cp_size, int)
                or self.encoder_cp_size < 1):
            raise ValueError(
                "accelerator.encoder_parallel.encoder_cp_size "
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
            raise ValueError("encoder_parallel must be a mapping")
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown encoder_parallel keys: {sorted(unknown)}")
        return cls(**raw)
