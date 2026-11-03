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
"""Reversible K3 per-expert and packed-expert checkpoint layouts."""
# Native Torch model adapters follow the same exception as other HF model families.
# pylint: disable=forbidden-backend-import

from typing import Any

import torch

from hyper_parallel.components.checkpoint.weight_conversion import ConversionOps, WeightConverter


def _one_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Each explicit expert checkpoint source must occur exactly once")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expert checkpoint conversion expects tensors")
    return value


class StackExpertWeights(ConversionOps):
    """Stack explicitly ordered expert matrices without changing their values."""

    def convert(self, input_dict: dict[str, Any], source_patterns: list[str],
                target_patterns: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Stack ordered expert matrices along the global expert axis.

        Args:
            input_dict: Source-pattern keys mapped to matrices or singleton lists.
            source_patterns: Explicit expert keys in numerical expert order.
            target_patterns: Exactly one key for the packed output tensor.
            **kwargs: Unused conversion-protocol options.

        Returns:
            The target key mapped to a tensor shaped [experts, output, input].

        Raises:
            ValueError: Sources are missing or matrices have incompatible shapes.
            TypeError: A source value is not a tensor.
        """
        del kwargs
        if len(target_patterns) != 1 or set(input_dict) != set(source_patterns):
            raise ValueError("Expert stacking requires every declared source and exactly one target")
        tensors = [_one_tensor(input_dict[name]) for name in source_patterns]
        if not tensors or any(value.ndim != 2 or value.shape != tensors[0].shape for value in tensors):
            raise ValueError("Expert sources must be nonempty, identically shaped matrices")
        return {target_patterns[0]: torch.stack(tensors, dim=0)}

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the inverse that restores the original per-expert FQNs."""
        return UnstackExpertWeights()


class UnstackExpertWeights(ConversionOps):
    """Restore explicitly ordered expert matrices from a complete logical tensor."""

    def convert(self, input_dict: dict[str, Any], source_patterns: list[str],
                target_patterns: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Split a complete logical expert tensor into ordered matrices.

        Args:
            input_dict: One source-pattern key mapped to the global packed tensor.
            source_patterns: Exactly one key for the packed source.
            target_patterns: Explicit per-expert keys in numerical expert order.
            **kwargs: Unused conversion-protocol options.

        Returns:
            Per-expert target keys mapped to unbound [output, input] matrices.

        Raises:
            ValueError: The source does not contain the complete declared expert axis.
            TypeError: The source value is not a tensor.
        """
        del kwargs
        if len(source_patterns) != 1 or set(input_dict) != set(source_patterns):
            raise ValueError("Expert unstacking requires exactly one complete source")
        tensor = _one_tensor(input_dict[source_patterns[0]])
        if tensor.ndim != 3 or tensor.shape[0] != len(target_patterns):
            raise ValueError("Expert unstacking requires the complete declared global expert axis")
        return dict(zip(target_patterns, tensor.unbind(0)))

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the forward stack operation."""
        return StackExpertWeights()


def build_expert_weight_converters(num_experts: int) -> list[WeightConverter]:
    """Declare reversible mappings between numbered and packed expert weights.

    Args:
        num_experts: Positive global expert count, before any EP partitioning.

    Returns:
        One stack/unstack converter for each of the w1, w2 and w3 matrices.

    Raises:
        ValueError: num_experts is not a positive integer.
    """
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts < 1:
        raise ValueError("num_experts must be a positive integer")
    return [WeightConverter(
        source_patterns=[rf"experts\.{index}\.{name}\.weight$" for index in range(num_experts)],
        target_patterns=f"experts.{name}", operations=[StackExpertWeights()],
    ) for name in ("w1", "w2", "w3")]
