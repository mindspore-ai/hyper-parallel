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
"""LLM-specific runtime batch processing and assembly."""

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

MODEL_INPUT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",
}
LOSS_INPUT_FIELDS = {
    "labels",
    "loss_mask",
    "stream_loss_mask",
}
CP_SEQUENCE_FIELDS = {
    "input_ids",
    "labels",
    "loss_mask",
    "stream_loss_mask",
    "position_ids",
    "seq_lens",
    "seq_lens_padded",
}


class LLMBatchProcessor:
    """Normalize, CP-shard, and classify fields for an LLM micro-batch."""

    def __init__(
            self,
            cp_sharder: ContextParallelBatchSharder,
            *,
            eod_token_id: int,
            reset_position_ids: bool,
            reset_attention_mask: bool,
            eod_mask_loss: bool,
            create_attention_mask: bool,
    ) -> None:
        """Store Dataset mask policy and the shared CP sharding capability."""
        self.cp_sharder = cp_sharder
        self.eod_token_id = eod_token_id
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.create_attention_mask = create_attention_mask

    @staticmethod
    def normalize_source_batch(
            source_batch: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Normalize ``tokens`` and HF ``input_ids`` at one boundary."""
        if source_batch is None:
            return None
        normalized_batch = dict(source_batch)
        if "input_ids" not in normalized_batch and "tokens" in normalized_batch:
            normalized_batch["input_ids"] = normalized_batch.pop("tokens")
        if "input_ids" not in normalized_batch:
            raise ValueError("LLM batch must contain 'input_ids' or 'tokens'")
        if "labels" not in normalized_batch:
            raise ValueError("LLM batch must contain 'labels'")
        if not platform.is_tensor(normalized_batch["labels"]):
            raise ValueError("LLM 'labels' must be a tensor")
        LLMBatchProcessor._validate_text_fields(normalized_batch)
        return normalized_batch

    @staticmethod
    def _validate_text_fields(batch: Mapping[str, Any]) -> None:
        """Validate the model-ready LLM tensor and sequence contract."""
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        if not platform.is_tensor(input_ids) or input_ids.ndim < 2:
            raise ValueError("LLM 'input_ids' must be a batched tensor")
        if not platform.is_tensor(labels) or tuple(labels.shape) != tuple(input_ids.shape):
            raise ValueError("LLM 'labels' must be a tensor with the same shape as 'input_ids'")

    def prepare_batch(self, batch: Mapping[str, Any]) -> PreparedBatch:
        """Apply text CP sharding and separate model, loss, and metadata fields."""
        runtime_batch = self._build_runtime_fields(batch)
        sharded_batch = self.cp_sharder.shard(runtime_batch, CP_SEQUENCE_FIELDS)
        model_inputs = {
            field_name: field_value
            for field_name, field_value in sharded_batch.items()
            if field_name in MODEL_INPUT_FIELDS
        }
        loss_inputs = {
            field_name: field_value
            for field_name, field_value in sharded_batch.items()
            if field_name in LOSS_INPUT_FIELDS
        }
        metadata = {
            field_name: field_value
            for field_name, field_value in sharded_batch.items()
            if field_name not in MODEL_INPUT_FIELDS and field_name not in LOSS_INPUT_FIELDS
        }
        prepared_batch = PreparedBatch(
            model_inputs=model_inputs,
            loss_inputs=loss_inputs,
            metadata=metadata,
        )
        return prepared_batch

    def _build_runtime_fields(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Rebuild masks and positions after TP token transport."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = labels >= 0
        if self.eod_mask_loss:
            loss_mask = loss_mask & (input_ids != self.eod_token_id)

        runtime_batch = {
            "input_ids": abs(input_ids),
            "labels": abs(labels),
            "loss_mask": loss_mask,
        }
        for field_name in ("seq_lens", "seq_lens_padded"):
            if field_name in batch:
                runtime_batch[field_name] = batch[field_name]
        attention_mask, position_ids = self._build_attention_mask_and_positions(input_ids)
        runtime_batch["position_ids"] = position_ids
        if attention_mask is not None:
            runtime_batch["attention_mask"] = attention_mask
        return runtime_batch

    def _build_attention_mask_and_positions(self, input_ids: Any) -> tuple[Any | None, Any]:
        """Build causal fields, including EOD resets, with tensor operations."""
        batch_size = int(input_ids.shape[0])
        sequence_length = int(input_ids.shape[-1])
        device = getattr(input_ids, "device", None)
        token_positions = platform.arange(
            sequence_length,
            dtype=input_ids.dtype,
            device=device,
        )
        position_ids = platform.zeros(
            (batch_size, 1),
            dtype=input_ids.dtype,
            device=device,
        ) + token_positions.reshape(1, sequence_length)

        segment_ids = None
        if self.reset_position_ids or self.reset_attention_mask:
            shifted_eod = platform.zeros(
                tuple(input_ids.shape),
                dtype=input_ids.dtype,
                device=device,
            )
            shifted_eod[:, 1:] = (input_ids[:, :-1] == self.eod_token_id)
            segment_ids = shifted_eod.cumsum(-1)

        query_positions = token_positions.reshape(1, sequence_length, 1)
        key_positions = token_positions.reshape(1, 1, sequence_length)
        causal_positions = key_positions <= query_positions
        if self.reset_position_ids:
            same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
            position_ids = (same_segment & causal_positions).sum(-1) - 1

        attention_mask = None
        if self.create_attention_mask:
            if self.reset_attention_mask:
                same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
                allowed_attention = same_segment & causal_positions
            else:
                allowed_attention = causal_positions
            attention_mask = (~allowed_attention).reshape(
                allowed_attention.shape[0],
                1,
                sequence_length,
                sequence_length,
            )
        return attention_mask, position_ids


class LLMGetBatch(RuntimeBatchAdapter):
    """LLM adapter assembled from common transport and LLM processing."""


def build_llm_get_batch(
        *,
        parallel_context: BatchParallelContext,
        device: Any,
        pipeline_router: PipelineBatchRouter | None = None,
        eod_token_id: int = 0,
        reset_position_ids: bool = False,
        reset_attention_mask: bool = False,
        eod_mask_loss: bool = False,
        create_attention_mask: bool = True,
) -> LLMGetBatch:
    """Build the LLM runtime batch adapter.

    Args:
        parallel_context: TP/CP topology and reserved PP routing metadata.
        device: Destination model device.
        pipeline_router: Optional stage-aware PP router. No default PP router is implemented.
        eod_token_id: Token that terminates one packed document.
        reset_position_ids: Whether positions restart after every EOD token.
        reset_attention_mask: Whether attention is isolated across EOD boundaries.
        eod_mask_loss: Whether EOD-token losses are excluded.
        create_attention_mask: Whether to materialize the causal attention mask.

    Returns:
        Callable LLM batch adapter.
    """
    transport = DistributedBatchTransport(
        parallel_context=parallel_context,
        device=device,
        field_names={"input_ids", "labels", "seq_lens", "seq_lens_padded"},
    )
    cp_sharder = ContextParallelBatchSharder(parallel_context)
    processor = LLMBatchProcessor(
        cp_sharder,
        eod_token_id=eod_token_id,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        eod_mask_loss=eod_mask_loss,
        create_attention_mask=create_attention_mask,
    )
    get_batch = LLMGetBatch(
        parallel_context=parallel_context,
        transport=transport,
        processor=processor,
        pipeline_router=pipeline_router,
    )
    return get_batch


__all__ = ["LLMBatchProcessor", "LLMGetBatch", "build_llm_get_batch"]
