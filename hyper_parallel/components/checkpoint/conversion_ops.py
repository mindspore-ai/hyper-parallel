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
"""Reusable checkpoint conversion operations."""

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from hyper_parallel.components.checkpoint.weight_conversion import ConversionOps


def _single_tensor(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Return the single tensor collected for one checkpoint pattern."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("expected exactly one tensor per checkpoint pattern")
        return value[0]
    return value


class ConcatenateWithSections(ConversionOps):
    """Concatenate tensors and retain their section sizes for reverse conversion."""

    def __init__(self, sections: tuple[int, ...], dim: int = 0) -> None:
        """Configure the tensor sizes along the concatenation dimension."""
        if len(sections) < 2 or any(not isinstance(size, int) or size <= 0 for size in sections):
            raise ValueError("sections must contain at least two positive integers")
        self.sections = tuple(sections)
        self.dim = dim

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Concatenate one tensor from each source pattern."""
        del kwargs
        if len(source_patterns) != len(self.sections) or len(target_patterns) != 1:
            raise ValueError(
                "ConcatenateWithSections requires one source per section and one target"
            )
        tensors = [_single_tensor(input_dict[pattern]) for pattern in source_patterns]
        actual_sections = tuple(tensor.shape[self.dim] for tensor in tensors)
        if actual_sections != self.sections:
            raise ValueError(
                f"source sizes {actual_sections} do not match configured sections {self.sections}"
            )
        return {target_patterns[0]: torch.cat(tensors, dim=self.dim).contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the section-based split used for reverse conversion."""
        return Split(self.sections, dim=self.dim)


class Split(ConversionOps):
    """Split a tensor into explicitly sized sections."""

    def __init__(self, sections: tuple[int, ...], dim: int = 0) -> None:
        """Configure the output sizes along the split dimension."""
        if len(sections) < 2 or any(not isinstance(size, int) or size <= 0 for size in sections):
            raise ValueError("sections must contain at least two positive integers")
        self.sections = tuple(sections)
        self.dim = dim

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Split one source tensor and assign each section to one target pattern."""
        del kwargs
        if len(source_patterns) != 1 or len(target_patterns) != len(self.sections):
            raise ValueError("Split requires one source and one target per section")
        tensor = _single_tensor(input_dict[source_patterns[0]])
        tensors = torch.split(tensor, self.sections, dim=self.dim)
        return {
            pattern: value.contiguous()
            for pattern, value in zip(target_patterns, tensors)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the section-aware concatenation used for reverse conversion."""
        return ConcatenateWithSections(self.sections, dim=self.dim)


class AddScalar(ConversionOps):
    """Add a scalar to one checkpoint tensor."""

    def __init__(self, value: float) -> None:
        """Configure the scalar added during conversion."""
        self.value = value

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Add the configured scalar and assign the converted tensor."""
        del kwargs
        if len(source_patterns) != 1 or len(target_patterns) != 1:
            raise ValueError("AddScalar requires exactly one source and one target")
        tensor = _single_tensor(input_dict[source_patterns[0]])
        return {target_patterns[0]: tensor + self.value}

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the inverse scalar addition used during export."""
        return AddScalar(-self.value)


class InterleaveQKV(ConversionOps):
    """Convert Q, K, and V rows to the GQA-interleaved layout."""

    def __init__(
        self,
        num_key_value_heads: int,
        num_key_value_groups: int,
        query_head_dim: int,
        value_head_dim: int,
        source_is_fused: bool,
    ) -> None:
        """Configure the source projection layout and GQA dimensions."""
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim
        self.source_is_fused = source_is_fused

    @property
    def _source_sections(self) -> tuple[int, int, int]:
        return (
            self.num_key_value_heads
            * self.num_key_value_groups
            * self.query_head_dim,
            self.num_key_value_heads * self.query_head_dim,
            self.num_key_value_heads * self.value_head_dim,
        )

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Convert fused or separate Q/K/V rows to the grouped layout."""
        del kwargs
        if len(target_patterns) != 1:
            raise ValueError("grouped GQA projection requires exactly one target")
        if self.source_is_fused:
            if len(source_patterns) != 1:
                raise ValueError("fused GQA projection requires exactly one source")
            fused = _single_tensor(input_dict[source_patterns[0]])
            query, key, value = torch.split(fused, self._source_sections, dim=0)
        else:
            if len(source_patterns) != 3:
                raise ValueError("separate GQA projection requires Q, K, and V sources")
            query, key, value = (
                _single_tensor(input_dict[pattern]) for pattern in source_patterns
            )

        trailing_shape = query.shape[1:]
        query = query.reshape(
            self.num_key_value_heads,
            self.num_key_value_groups * self.query_head_dim,
            *trailing_shape,
        )
        key = key.reshape(
            self.num_key_value_heads,
            self.query_head_dim,
            *trailing_shape,
        )
        value = value.reshape(
            self.num_key_value_heads,
            self.value_head_dim,
            *trailing_shape,
        )
        grouped = torch.cat((query, key, value), dim=1).flatten(0, 1)
        return {target_patterns[0]: grouped.contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the conversion that restores the source Q/K/V layout."""
        return DeinterleaveQKV(
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.query_head_dim,
            self.value_head_dim,
            self.source_is_fused,
        )


class DeinterleaveQKV(ConversionOps):
    """Restore concatenated or separate Q, K, and V rows."""

    def __init__(
        self,
        num_key_value_heads: int,
        num_key_value_groups: int,
        query_head_dim: int,
        value_head_dim: int,
        target_is_fused: bool,
    ) -> None:
        """Configure the grouped layout and restored projection form."""
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim
        self.target_is_fused = target_is_fused

    @property
    def _group_sections(self) -> tuple[int, int, int]:
        return (
            self.num_key_value_groups * self.query_head_dim,
            self.query_head_dim,
            self.value_head_dim,
        )

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Restore one grouped projection as fused or separate Q/K/V rows."""
        del kwargs
        if len(source_patterns) != 1:
            raise ValueError("ungrouping GQA projection requires exactly one source")
        grouped = _single_tensor(input_dict[source_patterns[0]])
        grouped = grouped.reshape(
            self.num_key_value_heads,
            sum(self._group_sections),
            *grouped.shape[1:],
        )
        query, key, value = torch.split(grouped, self._group_sections, dim=1)
        query = query.flatten(0, 1).contiguous()
        key = key.flatten(0, 1).contiguous()
        value = value.flatten(0, 1).contiguous()

        if self.target_is_fused:
            if len(target_patterns) != 1:
                raise ValueError("fused GQA projection requires exactly one target")
            return {target_patterns[0]: torch.cat((query, key, value), dim=0)}
        if len(target_patterns) != 3:
            raise ValueError("separate GQA projection requires Q, K, and V targets")
        return dict(zip(target_patterns, (query, key, value)))

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the conversion that groups Q/K/V rows by KV head."""
        return InterleaveQKV(
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.query_head_dim,
            self.value_head_dim,
            self.target_is_fused,
        )


class InterleaveGateQKV(ConversionOps):
    """Group per-query-head Q/gate plus K/V rows into one projection."""

    def __init__(
        self,
        num_key_value_heads: int,
        num_key_value_groups: int,
        query_head_dim: int,
        value_head_dim: int,
    ) -> None:
        """Configure the gated query layout and GQA dimensions."""
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Convert interleaved Q/gate plus K/V into grouped Q/K/V/gate rows."""
        del kwargs
        if len(source_patterns) != 3 or len(target_patterns) != 1:
            raise ValueError(
                "gated GQA conversion requires Q/gate, K, V sources and one target"
            )
        query_gate, key, value = (
            _single_tensor(input_dict[pattern]) for pattern in source_patterns
        )
        trailing_shape = query_gate.shape[1:]
        query_gate = query_gate.reshape(
            self.num_key_value_heads,
            self.num_key_value_groups,
            2,
            self.query_head_dim,
            *trailing_shape,
        )
        query = query_gate[:, :, 0].reshape(
            self.num_key_value_heads,
            self.num_key_value_groups * self.query_head_dim,
            *trailing_shape,
        )
        gate = query_gate[:, :, 1].reshape(
            self.num_key_value_heads,
            self.num_key_value_groups * self.query_head_dim,
            *trailing_shape,
        )
        key = key.reshape(
            self.num_key_value_heads,
            self.query_head_dim,
            *trailing_shape,
        )
        value = value.reshape(
            self.num_key_value_heads,
            self.value_head_dim,
            *trailing_shape,
        )
        grouped = torch.cat((query, key, value, gate), dim=1).flatten(0, 1)
        return {target_patterns[0]: grouped.contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the conversion that restores per-head Q/gate rows and K/V."""
        return DeinterleaveGateQKV(
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.query_head_dim,
            self.value_head_dim,
        )


class DeinterleaveGateQKV(ConversionOps):
    """Restore per-query-head Q/gate rows and separate K/V rows."""

    def __init__(
        self,
        num_key_value_heads: int,
        num_key_value_groups: int,
        query_head_dim: int,
        value_head_dim: int,
    ) -> None:
        """Configure the grouped gated-GQA layout."""
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim

    @property
    def _group_sections(self) -> tuple[int, int, int, int]:
        return (
            self.num_key_value_groups * self.query_head_dim,
            self.query_head_dim,
            self.value_head_dim,
            self.num_key_value_groups * self.query_head_dim,
        )

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Restore Q/gate, K, and V rows from one grouped projection."""
        del kwargs
        if len(source_patterns) != 1 or len(target_patterns) != 3:
            raise ValueError(
                "gated GQA reverse conversion requires one source and Q/gate, K, V targets"
            )
        grouped = _single_tensor(input_dict[source_patterns[0]])
        trailing_shape = grouped.shape[1:]
        grouped = grouped.reshape(
            self.num_key_value_heads,
            sum(self._group_sections),
            *trailing_shape,
        )
        query, key, value, gate = torch.split(
            grouped, self._group_sections, dim=1
        )
        query = query.reshape(
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.query_head_dim,
            *trailing_shape,
        )
        gate = gate.reshape(
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.query_head_dim,
            *trailing_shape,
        )
        query_gate = torch.stack((query, gate), dim=2).flatten(0, 3)
        return {
            target_patterns[0]: query_gate.contiguous(),
            target_patterns[1]: key.flatten(0, 1).contiguous(),
            target_patterns[2]: value.flatten(0, 1).contiguous(),
        }

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the conversion that groups Q/K/V and extracts gate rows."""
        return InterleaveGateQKV(
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.query_head_dim,
            self.value_head_dim,
        )
