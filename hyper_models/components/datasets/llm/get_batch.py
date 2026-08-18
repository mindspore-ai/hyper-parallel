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
from dataclasses import replace
from typing import Any

from hyper_parallel.platform import get_platform
from hyper_models.components.datasets.batch_adapter import RuntimeBatchAdapter
from hyper_models.components.datasets.parallel.batch_context import (
    BatchParallelContext,
    create_batch_parallel_context,
)
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
_PACKED_LENGTH_FIELDS = {"seq_lens", "seq_lens_padded"}
_PACKED_LENGTH_SENTINEL = -1000


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

    def prepare_batch(self, batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply text CP sharding and separate model inputs from loss inputs."""
        runtime_batch = self.build_runtime_batch(batch)
        sharded_batch = self.cp_sharder.shard(runtime_batch, CP_SEQUENCE_FIELDS)
        return self.prepare_runtime_batch(sharded_batch)

    def build_runtime_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Build complete runtime fields before any CP sequence distribution."""
        return self._build_runtime_fields(batch)

    @staticmethod
    def prepare_runtime_batch(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Classify an already CP-local runtime batch for model execution."""
        model_inputs = {field_name: field_value for field_name, field_value in batch.items()
                        if field_name in MODEL_INPUT_FIELDS}
        loss_inputs = {field_name: field_value for field_name, field_value in batch.items()
                       if field_name in LOSS_INPUT_FIELDS}
        return model_inputs, loss_inputs

    def _build_runtime_fields(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Rebuild masks and positions after TP token transport."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"] != 0 if "loss_mask" in batch else labels >= 0
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


class ContextParallelBatchDistributor:
    """Scatter one complete LLM runtime batch from CP rank zero."""

    def __init__(
            self,
            parallel_context: BatchParallelContext,
            device: Any,
    ) -> None:
        """Store the CP topology and destination device.

        Args:
            parallel_context: Runtime topology for the current DP replica.
            device: Destination model device.

        Raises:
            ValueError: If CP is enabled without a CP process group.
        """
        if parallel_context.cp_size > 1 and parallel_context.cp_group is None:
            raise ValueError("CP batch distribution requires a CP process group")
        self.parallel_context = parallel_context
        self.device = device

    def scatter_batch(
            self,
            source_batch: Mapping[str, Any] | None,
            *,
            source_exhausted: bool,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Scatter one CP-local sequence shard to every CP rank.

        Only CP rank zero supplies ``source_batch``. All ranks participate in
        schema exchange and tensor scatters so dynamically shaped packed batches
        remain supported.

        Args:
            source_batch: Complete runtime batch on CP rank zero.
            source_exhausted: Whether the source DataLoader is exhausted.

        Returns:
            Current CP rank's device batch and the synchronized exhaustion flag.
        """
        if self.parallel_context.cp_size <= 1:
            if source_exhausted:
                return None, True
            if source_batch is None:
                raise ValueError("source batch is required when CP is disabled")
            return self._move_batch_to_device(source_batch), False

        source_shards = self._build_source_shards(source_batch, source_exhausted)
        source_payload = self._exchange_source_payload(source_shards, source_exhausted)
        if source_payload.get("exhausted", False):
            return None, True

        local_batch = dict(source_payload["object_values"][self.parallel_context.cp_rank])
        tensor_schemas = source_payload["tensor_schemas"]
        for field_name, field_schema in tensor_schemas.items():
            output = platform.empty(
                field_schema["shape"],
                dtype=field_schema["dtype"],
                device=self.device,
            )
            scatter_list = self._build_scatter_list(source_shards, field_name, field_schema["shape"])
            platform.scatter(
                output,
                scatter_list,
                group=self.parallel_context.cp_group,
                group_src=0,
            )
            local_batch[field_name] = output
        return local_batch, False

    def _build_source_shards(
            self,
            source_batch: Mapping[str, Any] | None,
            source_exhausted: bool,
    ) -> list[dict[str, Any]] | None:
        """Build every CP shard only on CP rank zero."""
        if self.parallel_context.cp_rank != 0:
            return None
        if source_exhausted:
            return None
        if source_batch is None:
            raise ValueError("CP rank zero must provide the complete source batch")

        source_shards = []
        for cp_rank in range(self.parallel_context.cp_size):
            rank_context = replace(self.parallel_context, cp_rank=cp_rank)
            rank_sharder = ContextParallelBatchSharder(rank_context)
            source_shards.append(rank_sharder.shard(source_batch, CP_SEQUENCE_FIELDS))
        return source_shards

    def _exchange_source_payload(
            self,
            source_shards: list[dict[str, Any]] | None,
            source_exhausted: bool,
    ) -> dict[str, Any]:
        """Exchange dynamic tensor schemas and rank-local object values."""
        payload = None
        if self.parallel_context.cp_rank == 0:
            if source_exhausted:
                payload = {"exhausted": True}
            else:
                if source_shards is None:
                    raise ValueError("CP rank zero must build source shards")
                payload = {
                    "exhausted": False,
                    "tensor_schemas": self._build_tensor_schemas(source_shards),
                    "object_values": [
                        {
                            field_name: field_value
                            for field_name, field_value in shard.items()
                            if not platform.is_tensor(field_value)
                        }
                        for shard in source_shards
                    ],
                }

        gathered_payloads = [None] * self.parallel_context.cp_size
        platform.all_gather_object(
            gathered_payloads,
            payload,
            group=self.parallel_context.cp_group,
        )
        source_payload = gathered_payloads[0]
        if not isinstance(source_payload, dict):
            raise ValueError("CP batch source rank did not provide batch metadata")
        return source_payload

    @staticmethod
    def _build_tensor_schemas(source_shards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Build common schemas, padding packed-length metadata when needed."""
        tensor_schemas = {}
        for field_name, field_value in source_shards[0].items():
            if not platform.is_tensor(field_value):
                continue
            field_values = [shard[field_name] for shard in source_shards]
            prefix_shape = tuple(field_value.shape[:-1])
            if any(tuple(value.shape[:-1]) != prefix_shape for value in field_values):
                raise ValueError(f"CP field {field_name!r} must have matching non-sequence dimensions")
            last_dim = max(int(value.shape[-1]) for value in field_values)
            if any(int(value.shape[-1]) != last_dim for value in field_values):
                if field_name not in _PACKED_LENGTH_FIELDS:
                    raise ValueError(f"CP field {field_name!r} produced unequal shard shapes")
            tensor_schemas[field_name] = {
                "shape": prefix_shape + (last_dim,),
                "dtype": field_value.dtype,
            }
        return tensor_schemas

    def _build_scatter_list(
            self,
            source_shards: list[dict[str, Any]] | None,
            field_name: str,
            output_shape: tuple[int, ...],
    ) -> list[Any] | None:
        """Materialize equal-shaped device tensors for one scatter call."""
        if self.parallel_context.cp_rank != 0:
            return None
        if source_shards is None:
            raise ValueError("CP rank zero must provide source shards")

        if field_name not in CP_SEQUENCE_FIELDS:
            field_value = platform.move_to_device(
                source_shards[0][field_name],
                self.device,
                non_blocking=True,
            )
            return [field_value] * self.parallel_context.cp_size

        scatter_list = []
        for shard in source_shards:
            field_value = shard[field_name]
            if tuple(field_value.shape) != output_shape:
                padded_value = platform.zeros(
                    output_shape,
                    dtype=field_value.dtype,
                    device=getattr(field_value, "device", None),
                )
                padded_value[...] = _PACKED_LENGTH_SENTINEL
                padded_value[..., :field_value.shape[-1]] = field_value
                field_value = padded_value
            scatter_list.append(
                platform.move_to_device(field_value, self.device, non_blocking=True)
            )
        return scatter_list

    def _move_batch_to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Move top-level tensors while retaining Python metadata."""
        device_batch = {}
        for field_name, field_value in batch.items():
            if platform.is_tensor(field_value):
                device_batch[field_name] = platform.move_to_device(
                    field_value,
                    self.device,
                    non_blocking=True,
                )
            else:
                device_batch[field_name] = field_value
        return device_batch


class LLMGetBatch(RuntimeBatchAdapter):
    """LLM adapter with one DataLoader source and CP-first distribution."""

    def __init__(
            self,
            *,
            parallel_context: BatchParallelContext,
            transport: DistributedBatchTransport,
            processor: LLMBatchProcessor,
            cp_distributor: ContextParallelBatchDistributor,
            pipeline_router: PipelineBatchRouter | None = None,
    ) -> None:
        """Store the runtime components used to prepare one LLM batch."""
        super().__init__(
            parallel_context=parallel_context,
            transport=transport,
            processor=processor,
            pipeline_router=pipeline_router,
        )
        self.cp_distributor = cp_distributor

    def __call__(
            self,
            data_iterator: Any,
            *,
            external_batch: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read once per DP replica, CP-scatter, then TP-broadcast one batch."""
        if self.parallel_context.cp_size <= 1:
            return super().__call__(data_iterator, external_batch=external_batch)

        source_batch = None
        source_exhausted = False
        if self.parallel_context.cp_rank == 0:
            source_batch, source_exhausted = self.transport.read_source_batch(
                data_iterator,
                external_batch,
            )
            source_batch = self.processor.normalize_source_batch(source_batch)
            if self.pipeline_router is not None:
                source_batch, source_exhausted = self.pipeline_router.route_source_batch(
                    source_batch,
                    source_exhausted=source_exhausted,
                )

        # cp
        cp_local_batch = None
        cp_local_exhausted = False
        if self.parallel_context.tp_rank == 0:
            runtime_batch = (
                self.processor.build_runtime_batch(source_batch)
                if source_batch is not None
                else None
            )
            cp_local_batch, cp_local_exhausted = self.cp_distributor.scatter_batch(
                runtime_batch,
                source_exhausted=source_exhausted,
            )

        # tp broadcast
        distributed_batch = self.transport.broadcast_batch(
            cp_local_batch,
            source_exhausted=cp_local_exhausted,
        )
        return self.processor.prepare_runtime_batch(distributed_batch)


def build_llm_get_batch(
        *,
        mesh_context: Any,
        device: Any,
        tokenizer: Any,
        data_config: Mapping[str, Any],
        pp_shared_data: bool = False,
        pipeline_router: PipelineBatchRouter | None = None,
) -> LLMGetBatch:
    """Build the LLM runtime batch adapter.

    Args:
        mesh_context: Model-parallel mesh used to derive the batch topology.
        device: Destination model device.
        tokenizer: Tokenizer that defines the end-of-document token.
        data_config: Dataset mask and position-ID policy.
        pp_shared_data: Whether pipeline stages share the source batch.
        pipeline_router: Optional stage-aware PP router. No default PP router is implemented.

    Returns:
        Callable LLM batch adapter.
    """
    eod_token_id = getattr(tokenizer, "eod", getattr(tokenizer, "eos_token_id", 0))
    parallel_context = create_batch_parallel_context(mesh_context, pp_shared_data=pp_shared_data)

    transport_fields = MODEL_INPUT_FIELDS | LOSS_INPUT_FIELDS | CP_SEQUENCE_FIELDS
    transport = DistributedBatchTransport(parallel_context=parallel_context, device=device, field_names=transport_fields)
    cp_sharder = ContextParallelBatchSharder(parallel_context)
    cp_distributor = ContextParallelBatchDistributor(parallel_context, device)
    processor = LLMBatchProcessor(
        cp_sharder,
        eod_token_id=eod_token_id,
        reset_position_ids=bool(data_config.get("reset_position_ids", False)),
        reset_attention_mask=bool(data_config.get("reset_attention_mask", False)),
        eod_mask_loss=bool(data_config.get("eod_mask_loss", False)),
        create_attention_mask=bool(data_config.get("create_attention_mask_in_dataloader", True)),
    )
    get_batch = LLMGetBatch(parallel_context=parallel_context, transport=transport, processor=processor,
                            cp_distributor=cp_distributor, pipeline_router=pipeline_router)
    return get_batch


__all__ = ["LLMBatchProcessor", "LLMGetBatch", "build_llm_get_batch"]
