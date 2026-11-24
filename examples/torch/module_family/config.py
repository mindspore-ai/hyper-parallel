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
"""Logical configuration owned by the module-family decoder example."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderConfig:
    """Global algorithm dimensions independent of a parallel job."""

    hidden_size: int = 8
    num_heads: int = 2
    num_experts: int = 4
    intermediate_size: int = 16
    top_k: int = 2
    model_type: str = "module_family_decoder"
    architectures: tuple[str, ...] = ("FamilyDecoder",)

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_heads", "num_experts", "intermediate_size", "top_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads or self.top_k > self.num_experts:
            raise ValueError("heads must divide hidden size and top_k cannot exceed num_experts")
