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

    Text fields use default collation. Modality fields such as
    ``pixel_values`` and ``image_grid_thw`` are concatenated along dim 0 so
    variable-length images batch correctly. This temporary implementation does
    not depend on the LLM batching pipeline.
    """

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

        batch = default_collate(text_samples)
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
    if packing:
        raise NotImplementedError("The temporary VLM collator does not support packing")
    if pad_token_id != 0 or ignore_index != IGNORE_INDEX or pad_to_length is not None:
        raise NotImplementedError("The temporary VLM collator does not support custom text padding")
    return VLMCollator()


__all__ = ["VLMCollator", "build_vlm_collator"]
