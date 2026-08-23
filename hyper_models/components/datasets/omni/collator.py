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
"""VLM micro-batch interfaces with deferred modality field policies."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from torch.utils.data import default_collate

from hyper_models.components.datasets.batching.batch import FieldCollateSpec

MetadataCollateFunc = Callable[[dict[str, Any], dict[str, int]], None]
TextCollateFunc = Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]]

TEXT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "loss_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
}

DEFAULT_MODAL_FIELD_SPECS = {
    "pixel_values": FieldCollateSpec("concat", 0),
    "pixel_values_videos": FieldCollateSpec("concat", 0),
    "input_features": FieldCollateSpec("concat", 0),
    "image_mask": FieldCollateSpec("pack", -1, 0),
    "video_mask": FieldCollateSpec("pack", -1, 0),
    "audio_mask": FieldCollateSpec("pack", -1, 0),
    "image_grid_hw": FieldCollateSpec("concat", 0),
    "image_grid_thw": FieldCollateSpec("concat", 0),
    "video_grid_thw": FieldCollateSpec("concat", 0),
    "audio_feature_lengths": FieldCollateSpec("concat", 0),
}


class VLMCollator:
    """Stack fixed-shape VLM samples into one model micro-batch.

    Text fields use PyTorch default collation. The active modality path
    uses fixed shapes and PyTorch default collation.
    Per-field packing and metadata generation remain explicit, deferred
    interfaces.

    Args:
        text_collate_fn: Collation callable used for shared text fields.
        modal_field_specs: Reserved modality field policies.
        metadata_collate_func: Reserved model-provided metadata hook.
    """

    def __init__(
            self,
            text_collate_fn: TextCollateFunc,
            *,
            modal_field_specs: Optional[Mapping[str, FieldCollateSpec]] = None,
            metadata_collate_func: Optional[MetadataCollateFunc] = None,
    ) -> None:
        self.text_collate_fn = text_collate_fn
        self.modal_field_specs = (
            dict(modal_field_specs)
            if modal_field_specs is not None
            else None
        )
        self.metadata_collate_func = metadata_collate_func

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Collate text and modality fields into one micro-batch.

        Args:
            samples: Omni samples selected for one micro-batch.

        Returns:
            Fixed-shape text and modality model inputs.

        Raises:
            NotImplementedError: If field-specific modality collation or
                metadata generation is requested.
        """
        if self.modal_field_specs is not None:
            raise NotImplementedError("VLM field-specific collation is not implemented yet")

        if self.metadata_collate_func is not None:
            raise NotImplementedError("VLM metadata collation is not implemented yet")

        text_samples = [
            {field: value for field, value in sample.items() if field in TEXT_FIELDS}
            for sample in samples
        ]
        modal_samples = [
            {field: value for field, value in sample.items() if field not in TEXT_FIELDS}
            for sample in samples
        ]

        batch = self.text_collate_fn(text_samples)
        if any(modal_sample for modal_sample in modal_samples):
            modal_batch = default_collate(modal_samples)
            batch.update(modal_batch)
        return batch


OmniCollator = VLMCollator


def build_vlm_collator(
        *,
        modal_field_specs: Optional[Mapping[str, FieldCollateSpec]] = None,
        metadata_collate_func: Optional[MetadataCollateFunc] = None,
) -> VLMCollator:
    """Build the fixed-shape VLM micro-batch collator.

    Args:
        modal_field_specs: Reserved modality field policies.
        metadata_collate_func: Reserved model-provided metadata hook.

    Returns:
        A collator producing one VLM micro-batch dictionary.
    """
    text_collate_fn = default_collate
    collator = VLMCollator(
        text_collate_fn=text_collate_fn,
        modal_field_specs=modal_field_specs,
        metadata_collate_func=metadata_collate_func,
    )
    return collator


def build_omni_collator(
        *,
        modal_field_specs: Optional[Mapping[str, FieldCollateSpec]] = None,
        metadata_collate_func: Optional[MetadataCollateFunc] = None,
) -> VLMCollator:
    """Build the VLM collator through the existing Omni entry point."""
    vlm_collator = build_vlm_collator(
        modal_field_specs=modal_field_specs,
        metadata_collate_func=metadata_collate_func,
    )
    return vlm_collator
