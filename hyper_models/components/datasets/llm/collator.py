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
"""PanGu-style LLM micro-batch collation with reserved packing interfaces."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import default_collate

from hyper_models.components.datasets.build_collate_fn import build_collate_fn
from hyper_models.components.datasets.collator import FieldCollateSpec
from hyper_models.components.utils.constants import IGNORE_INDEX


class LLMCollator:
    """Stack fixed-length text samples into one model micro-batch.

    The active path follows PanGu: Dataset and data transform produce complete
    fixed-length model samples, and the collator stacks them without changing
    labels, masks, position IDs, or sample order. VeOmni-style packing and
    field-specific policies are represented by constructor arguments but are
    intentionally deferred.

    Args:
        packing: Reserved switch for VeOmni-style sequence packing.
        field_specs: Reserved field-specific collation policies.
        pad_token_id: Reserved padding value for packed input IDs.
        ignore_index: Reserved label value excluded from packed loss.
        pad_to_length: Reserved packed sequence length.
    """

    def __init__(
            self,
            *,
            packing: bool = False,
            field_specs: Mapping[str, FieldCollateSpec] | None = None,
            pad_token_id: int = 0,
            ignore_index: int = IGNORE_INDEX,
            pad_to_length: int | None = None,
    ) -> None:
        self.packing = packing
        self.field_specs = dict(field_specs) if field_specs is not None else None
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.pad_to_length = pad_to_length

    @staticmethod
    def _as_tensor(value: Any, field: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sample field {field!r} must be tensor-like") from exc

    def _normalize_sample(self, sample: Mapping[str, Any], index: int) -> dict[str, Any]:
        normalized = dict(sample)
        input_field = "input_ids" if "input_ids" in normalized else "tokens"
        if input_field not in normalized:
            raise ValueError(f"LLM sample {index} must contain 'input_ids' or 'tokens'")
        if "labels" not in normalized:
            raise ValueError(f"LLM sample {index} must contain 'labels'")

        input_ids = self._as_tensor(normalized[input_field], input_field)
        labels = self._as_tensor(normalized["labels"], "labels")
        if input_ids.shape[-1] != labels.shape[-1]:
            raise ValueError(
                f"LLM sample {index} input tokens and labels must have the same sequence length"
            )

        normalized[input_field] = input_ids
        normalized["labels"] = labels
        return normalized

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Collate LLM samples.

        Args:
            samples: Text samples for one micro-batch.

        Returns:
            Stacked fixed-length model inputs.

        Raises:
            NotImplementedError: If packing, padding, or field-specific
                collation is requested.
        """
        if self.packing:
            raise NotImplementedError("LLM packing is not implemented yet")
        if self.pad_to_length is not None:
            raise NotImplementedError("LLM dynamic padding is not implemented yet")
        if self.field_specs is not None:
            raise NotImplementedError("LLM field-specific collation is not implemented yet")

        normalized = [
            self._normalize_sample(sample, index)
            for index, sample in enumerate(samples)
        ]
        micro_batch = default_collate(normalized)
        return micro_batch


def build_llm_collator(
        *,
        packing: bool = False,
        field_specs: Mapping[str, FieldCollateSpec] | None = None,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
        pad_to_length: int | None = None,
) -> LLMCollator:
    """Build the PanGu-style LLM micro-batch collator.

    Args:
        packing: Reserved switch for VeOmni-style sequence packing.
        field_specs: Reserved field-specific collation policies.
        pad_token_id: Reserved padding value for input IDs.
        ignore_index: Reserved label value excluded from loss computation.
        pad_to_length: Reserved packed sequence length.

    Returns:
        A collator producing one model micro-batch dictionary.
    """
    internal_collator = LLMCollator(
        packing=packing,
        field_specs=field_specs,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        pad_to_length=pad_to_length,
    )
    collate_fn = build_collate_fn(
        internal_data_collator=internal_collator,
    )
    return collate_fn


__all__ = ["build_llm_collator"]
