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
"""Transport CPU image shards without staging complete images on a device."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.distributed as dist

from hyper_parallel.data.parallel import TPBatchBroadcaster
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderLocalInputs
from hyper_parallel.distributed.encoder_parallel.planning import DecoderSequenceLayout, build_segments
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelPlan, EncoderParallelRuntime


_METADATA_FIELD = "_encoder_transport_metadata"
_FORWARD_FIELDS = ("encoder_parallel_plan", "encoder_parallel_inputs")
_CHUNK_BYTES = 4 * 1024 * 1024


class EncoderParallelBatchTransport:
    """Plan from small metadata and send only each destination's CPU patches.

    Every decoder CP coordinate already has a complete CPU batch on TP0. TP
    peers may belong to different encoder DP buckets or CP shards, so their
    payloads are transported individually with bounded device staging.
    """

    def __init__(self, *, runtime: EncoderParallelRuntime, chunk_bytes: int = _CHUNK_BYTES) -> None:
        """Set the runtime and maximum device staging size in bytes."""
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
            raise ValueError("encoder transport chunk_bytes must be a positive integer")
        self.runtime = runtime
        self.chunk_bytes = chunk_bytes

    def _prepare_source(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Validate CPU images and keep tensor payloads out of TP broadcasting."""
        bridge = self.runtime.bridge
        prepared = {name: value for name, value in batch.items() if name not in bridge.input_fields}
        error = None
        items = ()
        dtype = torch.float32
        try:
            if set(batch).intersection((_METADATA_FIELD, *_FORWARD_FIELDS)):
                raise ValueError("encoder transport fields cannot be supplied by the DataLoader")
            for name in bridge.input_fields:
                value = batch.get(name)
                if torch.is_tensor(value) and value.device.type != "cpu":
                    raise ValueError(f"encoder source field {name} must remain on the CPU")
            items = bridge.parse_images(batch)
            pixels = batch.get(bridge.payload_field)
            if pixels is not None:
                dtype = pixels.dtype
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        prepared[_METADATA_FIELD] = (items, dtype, error)
        return prepared

    def _check_source_errors(self, error: str | None, device: Any) -> None:
        """Make all FSDP participants fail before plan or payload collectives."""
        invalid = torch.tensor(int(error is not None), device=device)
        if dist.is_initialized() and self.runtime.global_mesh.size() > 1:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.runtime.global_mesh.get_group())
        if bool(invalid.item()):
            raise ValueError(f"invalid encoder CPU input: {error or 'another source rank rejected its batch'}")

    def __call__(self, batch: Mapping[str, Any] | None,
                 broadcaster: TPBatchBroadcaster) -> tuple[dict[str, Any], dict[str, Any]]:
        """Broadcast ordinary inputs, agree on a plan, and materialize local patches."""
        source = self._prepare_source(batch) if batch is not None else None
        device_batch = broadcaster.broadcast(source, broadcast_all_fields=True)
        items, dtype, error = device_batch.pop(_METADATA_FIELD)
        self._check_source_errors(error, broadcaster.device)
        layout = DecoderSequenceLayout.build(
            mesh_context=self.runtime.mesh_context, cp_local_length=int(device_batch["input_ids"].shape[1]),
        )
        plan = self.runtime.finalize_plan(items, layout)
        inputs = {"encoder_parallel_plan": plan}
        if plan.active:
            inputs["encoder_parallel_inputs"] = EncoderLocalInputs(
                self._transport_patches(batch, plan, dtype, broadcaster),
            )
        return device_batch, inputs

    def _transport_patches(self, batch: Mapping[str, Any] | None, plan: EncoderParallelPlan,
                           dtype: torch.dtype, broadcaster: TPBatchBroadcaster) -> torch.Tensor:
        """Pack on the CPU and reuse a chunk-sized buffer across TP destinations."""
        context = broadcaster.parallel_context
        local_count = sum(segment.local_padded_length for segment in plan.local_segments)
        local = torch.empty((local_count, *self.runtime.bridge.empty_payload_shape),
                            dtype=dtype, device=broadcaster.device)
        chunk_elements = max(1, self.chunk_bytes // local.element_size())
        target_segments = []
        for target in range(context.tp_world_size):
            parent_rank = context.cp_rank * context.tp_world_size + target
            bucket, cp_rank = divmod(parent_rank, self.runtime.config.encoder_cp_size)
            target_segments.append(build_segments(plan.buckets[bucket], cp_rank, self.runtime.config.encoder_cp_size))
        patch_elements = local[0].numel()
        max_elements = max((sum(segment.local_padded_length for segment in segments) * patch_elements
                            for segments in target_segments[1:]), default=0)
        staging = torch.empty(min(chunk_elements, max_elements), dtype=dtype, device=broadcaster.device)
        for target, segments in enumerate(target_segments):
            packed = self.runtime.bridge.pack_payload(batch, segments) if context.tp_rank == 0 else None
            if target == 0:
                if context.tp_rank == 0:
                    local.copy_(packed)
                continue
            count = sum(segment.local_padded_length for segment in segments)
            elements = count * patch_elements
            for start in range(0, elements, chunk_elements):
                length = min(chunk_elements, elements - start)
                buffer = (local.view(-1)[start:start + length] if context.tp_rank == target
                          else staging[:length])
                if context.tp_rank == 0:
                    buffer.copy_(packed.view(-1)[start:start + length])
                dist.broadcast(buffer, group=context.tp_group, group_src=0)
        return local
