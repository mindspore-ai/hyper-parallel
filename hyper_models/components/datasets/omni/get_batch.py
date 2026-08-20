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
"""Omni-specific runtime batch processing and assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.platform import get_platform
from hyper_models.components.datasets.batch import PreparedBatch
from hyper_models.components.datasets.batch_adapter import RuntimeBatchAdapter
from hyper_models.components.datasets.parallel.batch_context import BatchParallelContext
from hyper_models.components.datasets.parallel.batch_transport import DistributedBatchTransport
from hyper_models.components.datasets.parallel.cp_sharder import ContextParallelBatchSharder
from hyper_models.components.datasets.parallel.pipeline_router import PipelineBatchRouter

platform = get_platform()

TEXT_MODEL_INPUT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
}
MODAL_MODEL_INPUT_FIELDS = {
    "pixel_values",
    "pixel_values_videos",
    "input_features",
    "audio_features",
    "audio_attention_mask",
    "audio_feature_lengths",
    "image_mask",
    "video_mask",
    "audio_mask",
    "image_grid_hw",
    "image_grid_thw",
    "video_grid_thw",
    "video_timestamp",
}
LOSS_INPUT_FIELDS = {
    "labels",
    "loss_mask",
    "stream_loss_mask",
}
CP_TEXT_SEQUENCE_FIELDS = {
    "input_ids",
    "labels",
    "loss_mask",
    "stream_loss_mask",
    "position_ids",
    "text_position_ids",
    "seq_lens",
    "seq_lens_padded",
}


class OmniBatchProcessor:
    """Normalize and classify text, vision, video, and audio batch fields."""

    def __init__(self, cp_sharder: ContextParallelBatchSharder) -> None:
        """Store the shared CP sharding capability."""
        self.cp_sharder = cp_sharder

    @staticmethod
    def normalize_source_batch(
            source_batch: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Normalize supported Omni text aliases without invoking LLM logic."""
        if source_batch is None:
            return None
        normalized_batch = dict(source_batch)
        for source_field in ("tokens", "text"):
            if "input_ids" not in normalized_batch and source_field in normalized_batch:
                normalized_batch["input_ids"] = normalized_batch.pop(source_field)
        if "input_ids" not in normalized_batch:
            raise ValueError("Omni batch must contain 'input_ids', 'tokens', or 'text'")
        if "labels" not in normalized_batch:
            raise ValueError("Omni batch must contain 'labels'")
        if not platform.is_tensor(normalized_batch["labels"]):
            raise ValueError("Omni 'labels' must be a tensor")
        if "loss_mask" not in normalized_batch:
            normalized_batch["loss_mask"] = normalized_batch["labels"] >= 0
        OmniBatchProcessor._validate_text_fields(normalized_batch)
        return normalized_batch

    @staticmethod
    def _validate_text_fields(batch: Mapping[str, Any]) -> None:
        """Validate text tensors without imposing rules on modality tensors."""
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        if not platform.is_tensor(input_ids) or input_ids.ndim < 2:
            raise ValueError("Omni 'input_ids' must be a batched tensor")
        if not platform.is_tensor(labels) or tuple(labels.shape) != tuple(input_ids.shape):
            raise ValueError("Omni 'labels' must be a tensor with the same shape as 'input_ids'")
        loss_mask = batch.get("loss_mask")
        if not platform.is_tensor(loss_mask) or tuple(loss_mask.shape) != tuple(input_ids.shape):
            raise ValueError("Omni 'loss_mask' must be a tensor with the same shape as 'input_ids'")

    def prepare_batch(self, batch: Mapping[str, Any]) -> PreparedBatch:
        """CP-shard text while preserving modality fields for the Omni model."""
        sharded_batch = self.cp_sharder.shard(batch, CP_TEXT_SEQUENCE_FIELDS)
        model_field_names = TEXT_MODEL_INPUT_FIELDS | MODAL_MODEL_INPUT_FIELDS
        model_inputs = {
            field_name: field_value
            for field_name, field_value in sharded_batch.items()
            if field_name in model_field_names
        }
        loss_inputs = {
            field_name: field_value
            for field_name, field_value in sharded_batch.items()
            if field_name in LOSS_INPUT_FIELDS
        }
        metadata = {
            field_name: field_value
            for field_name, field_value in sharded_batch.items()
            if field_name not in model_field_names and field_name not in LOSS_INPUT_FIELDS
        }
        prepared_batch = PreparedBatch(
            model_inputs=model_inputs,
            loss_inputs=loss_inputs,
            metadata=metadata,
        )
        return prepared_batch


class OmniGetBatch(RuntimeBatchAdapter):
    """Omni adapter assembled from common transport and Omni processing."""


def build_omni_get_batch(
        *,
        parallel_context: BatchParallelContext,
        device: Any,
        pipeline_router: PipelineBatchRouter | None = None,
) -> OmniGetBatch:
    """Build the Omni runtime batch adapter.

    Args:
        parallel_context: TP/CP topology and reserved PP routing metadata.
        device: Destination model device.
        pipeline_router: Optional stage-aware PP router. No default PP router is implemented.

    Returns:
        Callable Omni batch adapter.
    """
    transport = DistributedBatchTransport(
        parallel_context=parallel_context,
        device=device,
    )
    cp_sharder = ContextParallelBatchSharder(parallel_context)
    processor = OmniBatchProcessor(cp_sharder)
    get_batch = OmniGetBatch(
        parallel_context=parallel_context,
        transport=transport,
        processor=processor,
        pipeline_router=pipeline_router,
    )
    return get_batch


__all__ = ["OmniBatchProcessor", "OmniGetBatch", "build_omni_get_batch"]
