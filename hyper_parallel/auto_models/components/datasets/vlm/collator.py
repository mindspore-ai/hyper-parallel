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
"""Build the VLM micro-batch collator."""

from typing import Any, Optional

import torch
from torch.utils.data import default_collate

from hyper_parallel.auto_models.components.datasets.build_collate_fn import build_collate_fn
from hyper_parallel.auto_models.components.datasets.llm.collator import LLMCollator
from hyper_parallel.auto_models.components.utils.constants import IGNORE_INDEX

_TEXT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "loss_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
    "mm_token_type_ids",
}


class VLMCollator:
    """Collate text and modality fields into one VLM micro-batch.

    Text fields are delegated to :class:`LLMCollator` (stacked). Modality fields
    (``pixel_values``, ``image_grid_thw``, ...) are concatenated along dim 0 so
    variable-length images batch correctly.

    Args:
        text_collator: LLM collator used for shared text fields.
    """

    def __init__(self, text_collator: LLMCollator) -> None:
        """Bind the LLM collator used for shared text fields."""
        self.text_collator = text_collator

    def __call__(self, samples: Any) -> dict[str, Any]:
        """Collate one micro-batch of VLM samples."""
        text_samples = [
            {field: value for field, value in sample.items() if field in _TEXT_FIELDS}
            for sample in samples
        ]
        modal_samples = [
            {field: value for field, value in sample.items() if field not in _TEXT_FIELDS}
            for sample in samples
        ]

        batch = self.text_collator(text_samples)
        if any(modal_samples):
            for field in {field for sample in modal_samples for field in sample}:
                values = [sample[field] for sample in modal_samples if field in sample]
                batch[field] = (
                    torch.cat(values, dim=0)
                    if isinstance(values[0], torch.Tensor)
                    else default_collate(values)
                )
        return batch


def build_vlm_collator(
        *,
        packing: bool = False,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
        pad_to_length: Optional[int] = None,
) -> VLMCollator:
    """Build the VLM micro-batch collator.

    Args:
        packing: Reserved switch for VeOmni-style text packing.
        pad_token_id: Reserved padding value for text input IDs.
        ignore_index: Reserved label value excluded from loss computation.
        pad_to_length: Reserved packed text sequence length.

    Returns:
        A collator producing one VLM micro-batch dictionary.
    """
    text_collator = LLMCollator(
        packing=packing,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        pad_to_length=pad_to_length,
    )
    return build_collate_fn(internal_data_collator=VLMCollator(text_collator=text_collator))


__all__ = ["VLMCollator", "build_vlm_collator"]
