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
"""Build the VLM runtime batch adapter."""

from typing import Any

from hyper_parallel.auto_models.components.datasets.batch_adapter import RuntimeBatchAdapter
from hyper_parallel.auto_models.components.datasets.parallel.batch_context import BatchParallelContext
from hyper_parallel.auto_models.components.datasets.parallel.batch_transport import DistributedBatchTransport
from hyper_parallel.auto_models.components.datasets.parallel.cp_sharder import ContextParallelBatchSharder
from hyper_parallel.auto_models.components.datasets.parallel.pipeline_router import PipelineBatchRouter

_MODEL_INPUT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
    "mm_token_type_ids",
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
_LOSS_INPUT_FIELDS = {"labels", "loss_mask", "stream_loss_mask"}
_CP_TEXT_FIELDS = {
    "input_ids",
    "labels",
    "loss_mask",
    "stream_loss_mask",
    "position_ids",
    "text_position_ids",
    "mm_token_type_ids",
    "seq_lens",
    "seq_lens_padded",
}


class VLMBatchProcessor:
    """Normalize and classify text, vision, video, and audio batch fields."""

    def __init__(self, cp_sharder: ContextParallelBatchSharder) -> None:
        """Bind the context-parallel batch sharder."""
        self.cp_sharder = cp_sharder

    def normalize_source_batch(self, source_batch: Any) -> Any:
        """Normalize one collated batch into the VLM field contract."""
        if source_batch is None:
            return None
        batch = dict(source_batch)
        if "loss_mask" not in batch:
            batch["loss_mask"] = batch["labels"] >= 0
        return batch

    def prepare_batch(self, batch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """CP-shard text while preserving modality fields for the VLM model."""
        sharded_batch = self.cp_sharder.shard(batch, _CP_TEXT_FIELDS)
        model_inputs = {field: value for field, value in sharded_batch.items() if field in _MODEL_INPUT_FIELDS}
        loss_inputs = {field: value for field, value in sharded_batch.items() if field in _LOSS_INPUT_FIELDS}
        return model_inputs, loss_inputs


class VLMGetBatch(RuntimeBatchAdapter):
    """VLM adapter assembled from common transport and VLM processing."""


def build_vlm_get_batch(
        *,
        parallel_context: BatchParallelContext,
        device: Any,
        pipeline_router: PipelineBatchRouter | None = None,
) -> VLMGetBatch:
    """Build the VLM runtime batch adapter.

    Args:
        parallel_context: TP/CP topology and reserved PP routing metadata.
        device: Destination model device.
        pipeline_router: Optional stage-aware PP router. No default PP router is implemented.

    Returns:
        Callable VLM batch adapter.
    """
    transport = DistributedBatchTransport(parallel_context=parallel_context, device=device)
    processor = VLMBatchProcessor(ContextParallelBatchSharder(parallel_context))
    return VLMGetBatch(
        parallel_context=parallel_context,
        transport=transport,
        processor=processor,
        pipeline_router=pipeline_router,
    )


__all__ = ["VLMBatchProcessor", "VLMGetBatch", "build_vlm_get_batch"]
