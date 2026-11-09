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
"""Declare K3 expert FQN mappings using shared reversible operations."""

from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.checkpoint.conversion_ops import (
    StackExpertWeights, UnstackExpertWeights, StackExpertFC1, UnstackExpertFC1,
)


def build_expert_weight_converters(num_experts: int, *, fused_fc1: bool = False) -> list[WeightConverter]:
    """Declare reversible mappings between numbered and packed expert weights.

    Args:
        num_experts: Positive global expert count, before any EP partitioning.
        fused_fc1: Combine gate/up storage as w13, retaining the original HF names on export.

    Returns:
        Reversible converters for w1/w2/w3, or for combined w13 and w2 storage.

    Raises:
        ValueError: num_experts is not a positive integer.
    """
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts < 1:
        raise ValueError("num_experts must be a positive integer")
    converters = [WeightConverter(
        source_patterns=[rf"experts\.{index}\.{name}\.weight$" for index in range(num_experts)],
        target_patterns=f"experts.{name}", operations=[StackExpertWeights()],
    ) for name in (("w2",) if fused_fc1 else ("w1", "w2", "w3"))]
    if fused_fc1:
        converters.insert(0, WeightConverter(
            source_patterns=[rf"experts\.{index}\.{name}\.weight$"
                             for index in range(num_experts) for name in ("w1", "w3")],
            target_patterns="experts.w13", operations=[StackExpertFC1()],
        ))
    return converters


__all__ = ["StackExpertWeights", "UnstackExpertWeights", "StackExpertFC1", "UnstackExpertFC1",
           "build_expert_weight_converters"]
