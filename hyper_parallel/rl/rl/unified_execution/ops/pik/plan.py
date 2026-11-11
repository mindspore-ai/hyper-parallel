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
"""Static arithmetic and layout contract for Level 3 PIK."""
from __future__ import annotations

from dataclasses import dataclass


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PikPlan:
    """Fix global leaves and the BMM input layout across training/inference.

    Leaf outputs and the final root are BF16. Every internal tree node is FP32.
    Changing num_leaves changes the numerical contract, not just performance.
    """

    num_leaves: int = 8
    leaf_layout: str = "strided"

    def __post_init__(self) -> None:
        """Reject unsupported numerical and physical layouts."""
        _positive(self.num_leaves, "num_leaves")
        if self.num_leaves & (self.num_leaves - 1):
            raise ValueError("num_leaves must be a power of two")
        if self.leaf_layout not in ("strided", "packed"):
            raise ValueError("leaf_layout must be strided or packed")

    def validate(self, global_k: int, world_size: int) -> None:
        """Require each rank to own a contiguous complete subtree."""
        _positive(global_k, "global_k")
        _positive(world_size, "world_size")
        if self.num_leaves % world_size:
            raise ValueError("TP must divide the fixed global leaf count")
        if global_k % self.num_leaves:
            raise ValueError("global K must be divisible by num_leaves")

    def contract(self) -> dict:
        """Return the serializable contract checked before data communication."""
        return {"num_leaves": self.num_leaves, "leaf_layout": self.leaf_layout,
                "provider": "torch_npu_level3_bmm", "leaf_dtype": "bfloat16",
                "tree_dtype": "float32", "root_dtype": "bfloat16",
                "bias": "bf16_after_root_cast", "tree": "balanced_adjacent_lower_left"}
