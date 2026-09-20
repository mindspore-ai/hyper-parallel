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
"""Select and combine samples for multimodal packing."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from hyper_parallel.data.constants import DEFAULT_FIELD_PACK_DIMS, PACKING_METADATA_FIELDS
from hyper_parallel.data.dataset_logging import get_dataset_logger

logger = get_dataset_logger(__name__)


class PackingSelector(ABC):
    """Define the pre-packing seam over prepared sample metadata."""

    @abstractmethod
    def get_sample_cost(self, sample: Mapping[str, Any]) -> int:
        """Return the scheduling cost retained with one candidate sample."""
        raise NotImplementedError

    @abstractmethod
    def select_samples_to_pack(
            self,
            samples: Sequence[Mapping[str, Any]],
            token_budget: int,
    ) -> Sequence[int]:
        """Return indices of candidates that form the next packed sequence."""
        raise NotImplementedError


class FirstFitPackingSelector(PackingSelector):
    """Select candidates in source order until the token budget is full."""

    def get_sample_cost(self, sample: Mapping[str, Any]) -> int:
        """Read standard packing metadata or derive cost from encoded input IDs."""
        sample_cost = sample.get("packing_length")
        if sample_cost is None:
            sample_cost = self._get_input_length(sample)

        resolved_cost = int(sample_cost)
        if resolved_cost <= 0:
            raise ValueError("Packing sample cost must be positive")

        return resolved_cost

    def select_samples_to_pack(
            self,
            samples: Sequence[Mapping[str, Any]],
            token_budget: int,
    ) -> Sequence[int]:
        """Select the next ordered first-fit group from the candidate buffer."""
        if not samples:
            raise ValueError("Packing candidate buffer is empty")

        selected_indices = []
        selected_cost = 0
        for sample_index, sample in enumerate(samples):
            sample_cost = self.get_sample_cost(sample)
            sample_fits = selected_cost == 0 or selected_cost + sample_cost <= token_budget
            if sample_fits:
                selected_indices.append(sample_index)
                selected_cost += sample_cost

        return selected_indices

    @staticmethod
    def _get_input_length(sample: Mapping[str, Any]) -> int:
        """Derive sequence length from a fully encoded sample."""
        if "input_ids" not in sample:
            raise ValueError("Packing samples must provide packing_length or input_ids")

        input_ids = sample["input_ids"]
        input_shape = getattr(input_ids, "shape", None)
        if input_shape is not None:
            input_length = int(input_shape[-1])
            return input_length

        if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
            raise TypeError("Packing input_ids must be a tensor or sequence")

        input_length = len(input_ids)
        if input_ids and isinstance(input_ids[0], Sequence):
            input_length = len(input_ids[0])
        return input_length


class SamplePacker:
    """Combine selected encoded samples into one packed sample."""

    def __init__(self, field_pack_dims: Mapping[str, int] | None = None) -> None:
        """Configure the concatenation dimension of model fields.

        Args:
            field_pack_dims: Optional additions or overrides to the default
                token-last and modality-first field rules.
        """
        self.field_pack_dims = dict(DEFAULT_FIELD_PACK_DIMS)
        if field_pack_dims is not None:
            self.field_pack_dims.update(field_pack_dims)

    def pack_selected_samples(
            self,
            samples: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Concatenate one selected group and emit its sequence boundaries."""
        if not samples:
            raise ValueError("samples must contain at least one encoded model sample")

        packed_sample = {}
        sample_fields = set()
        for sample in samples:
            sample_fields.update(sample)

        for field in sample_fields:
            if field in PACKING_METADATA_FIELDS:
                continue

            if field.endswith("_token_starts"):
                packed_sample[field] = self._pack_token_starts(field, samples)
                continue

            field_values = []
            for sample in samples:
                if field in sample:
                    field_values.append(sample[field])

            packed_value = self._pack_field(field, field_values)
            packed_sample[field] = packed_value

        packed_sample["cu_seq_lens"] = self._build_sequence_boundaries(samples)
        return packed_sample

    def _pack_field(self, field: str, values: Sequence[Any]) -> Any:
        """Pack one field using its configured or inferred leading dimension."""
        if not values:
            raise ValueError(f"Cannot pack empty field {field!r}")

        pack_dim = self.field_pack_dims.get(field, 0)
        if all(isinstance(value, torch.Tensor) for value in values):
            packed_value = torch.cat(values, dim=pack_dim)
            return packed_value

        if field in self.field_pack_dims:
            tensor_values = []
            for value in values:
                tensor_values.append(torch.as_tensor(value))
            packed_value = torch.cat(tensor_values, dim=pack_dim)
            return packed_value

        if all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in values):
            packed_items = []
            for value in values:
                packed_items.extend(value)
            return packed_items

        if len(values) == 1:
            return values[0]

        raise TypeError(f"Field {field!r} requires a model-specific SamplePacker rule")

    @staticmethod
    def _pack_token_starts(
            field: str,
            samples: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        """Move sample-local modality starts into the packed token coordinate."""
        packed_starts = []
        sequence_offset = 0
        for sample in samples:
            if field in sample:
                token_starts = torch.as_tensor(sample[field])
                packed_starts.append(token_starts + sequence_offset)

            input_ids = sample.get("input_ids")
            if input_ids is None:
                raise ValueError(f"Packing {field!r} requires input_ids in every sample")
            sequence_offset += int(torch.as_tensor(input_ids).shape[-1])

        if not packed_starts:
            raise ValueError(f"Cannot pack empty field {field!r}")

        packed_token_starts = torch.cat(packed_starts, dim=0)
        return packed_token_starts

    @staticmethod
    def _build_sequence_boundaries(
            samples: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        """Build cumulative text boundaries for packed attention isolation."""
        sequence_lengths = []
        input_ids_reference = None
        for sample in samples:
            if "input_ids" not in sample:
                raise ValueError("Every packed sample must contain input_ids")

            input_ids = sample["input_ids"]
            if input_ids_reference is None and isinstance(input_ids, torch.Tensor):
                input_ids_reference = input_ids
            input_shape = getattr(input_ids, "shape", None)
            if input_shape is not None:
                sequence_length = int(input_shape[-1])
            else:
                sequence_length = len(input_ids)
            if sequence_length <= 0:
                raise ValueError("Packed samples must contain at least one token")
            sequence_lengths.append(sequence_length)

        cumulative_lengths = [0]
        for sequence_length in sequence_lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + sequence_length)

        if input_ids_reference is None:
            sequence_boundaries = torch.tensor(cumulative_lengths, dtype=torch.int32)
        else:
            sequence_boundaries = input_ids_reference.new_tensor(
                cumulative_lengths,
                dtype=torch.int32,
            )
        return sequence_boundaries


__all__ = ["FirstFitPackingSelector", "PackingSelector", "SamplePacker"]
