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

from dataclasses import replace
from typing import Any, Mapping

import torch
import torch.distributed as dist

from hyper_parallel.data.parallel import TPBatchBroadcaster
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderLocalInputs
from hyper_parallel.distributed.encoder_parallel.mesh import collect_decoder_metadata
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
        except (ValueError, TypeError, KeyError, NotImplementedError) as exc:
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
        layout, layout_error = None, None
        try:
            layout = DecoderSequenceLayout.build(
                mesh_context=self.runtime.mesh_context, cp_local_length=int(device_batch["input_ids"].shape[1]),
            )
        except (ValueError, TypeError, KeyError) as exc:
            layout_error = str(exc)
        self._check_source_errors(layout_error, broadcaster.device)
        plan = self.runtime.finalize_plan(items, layout)
        inputs = {"encoder_parallel_plan": plan}
        if plan.active:
            inputs["encoder_parallel_inputs"] = EncoderLocalInputs(
                self._transport_patches(batch, plan, dtype, broadcaster),
            )
        return device_batch, inputs

    def _payload_dtype(self, plan: EncoderParallelPlan, dtype: torch.dtype, *, device: Any = None) -> torch.dtype:
        """Agree on payload dtype only when different decoder samples share resources."""
        runtime = self.runtime
        if not any(plan.buckets):
            # Synthetic-only inputs need one common dtype even if empty source bundles differ.
            return torch.float32
        decoder_size = runtime.mesh_context.cp_size * runtime.mesh_context.tp_size
        if runtime.parent_mesh.size() == decoder_size:
            return dtype
        produces = any(item.source_rank == runtime.parent_rank for bucket in plan.buckets for item in bucket)
        dtypes = collect_decoder_metadata(dtype if produces else None, decoder_size=decoder_size,
                                          parent_mesh=runtime.parent_mesh, device=device)
        selected = {value for value in dtypes if value is not None}
        if len(selected) != 1:
            raise ValueError("Independent encoder image payloads require one common dtype")
        return selected.pop()

    def _transport_patches(self, batch: Mapping[str, Any] | None, plan: EncoderParallelPlan,
                           dtype: torch.dtype, broadcaster: TPBatchBroadcaster) -> torch.Tensor:
        """Execute one patch plan, choosing local, TP or cross-parent transport per segment."""
        runtime, bridge = self.runtime, self.runtime.bridge
        dtype = self._payload_dtype(plan, dtype, device=broadcaster.device)
        shape = bridge.empty_payload_shape
        rows = sum(segment.local_padded_length for segment in plan.local_segments)
        local = torch.zeros((rows, *shape), dtype=dtype, device=broadcaster.device)
        chunk_elements = max(1, self.chunk_bytes // local.element_size())
        max_rows = max(1, max((sum((item.sequence_length + runtime.config.encoder_cp_size - 1)
                                  // runtime.config.encoder_cp_size for item in bucket)
                               for bucket in plan.buckets), default=0))
        staging = torch.empty(min(chunk_elements, max_rows * local[0].numel()),
                              dtype=dtype, device=broadcaster.device)
        context = broadcaster.parallel_context
        decoder_size = runtime.mesh_context.cp_size * runtime.mesh_context.tp_size
        for target in range(runtime.parent_mesh.size()):
            bucket, cp_rank = divmod(target, runtime.config.encoder_cp_size)
            segments = build_segments(plan.buckets[bucket], cp_rank, runtime.config.encoder_cp_size)
            transfers = []
            for segment in segments:
                if segment.is_dummy:
                    continue
                source = segment.item.source_rank
                if source // decoder_size == target // decoder_size:
                    # Every decoder CP coordinate has a CPU replica on its TP leader.
                    source = target // context.tp_world_size * context.tp_world_size
                if transfers and transfers[-1][0] == source:
                    transfers[-1][1].append(segment)
                else:
                    transfers.append((source, [segment]))
            for source, contiguous in transfers:
                self._transfer_segments(batch, contiguous, source, target, local, staging, broadcaster)
        return local

    def _transfer_segments(self, batch: Mapping[str, Any] | None, segments: list,
                          source: int, target: int, local: torch.Tensor, staging: torch.Tensor,
                          broadcaster: TPBatchBroadcaster) -> None:
        """Preserve bounded staging and deterministic order across different transport groups."""
        rank = self.runtime.parent_rank
        tp_size = broadcaster.parallel_context.tp_world_size
        in_tp = source // tp_size == target // tp_size
        if in_tp and rank // tp_size != target // tp_size:
            return
        if not in_tp and rank not in (source, target):
            return
        start_row = segments[0].packed_offset
        count = sum(segment.local_padded_length for segment in segments)
        packed = self._pack_source_segments(batch, segments) if rank == source else None
        destination = local[start_row:start_row + count].view(-1)
        if source == target:
            if rank == target:
                destination.copy_(packed)
            return
        elements = count * local[0].numel()
        members = None if in_tp else dist.get_process_group_ranks(self.runtime.parent_mesh.get_group())
        for start in range(0, elements, staging.numel()):
            length = min(staging.numel(), elements - start)
            buffer = destination[start:start + length] if rank == target else staging[:length]
            if rank == source:
                buffer.copy_(packed[start:start + length])
            if in_tp:
                dist.broadcast(buffer, group=broadcaster.parallel_context.tp_group, group_src=0)
            elif rank == source:
                dist.send(buffer, dst=members[target], group=self.runtime.parent_mesh.get_group())
            else:
                dist.recv(buffer, src=members[source], group=self.runtime.parent_mesh.get_group())

    def _pack_source_segments(self, batch: Mapping[str, Any] | None, segments: list) -> torch.Tensor:
        """Pack contiguous source intervals with offsets relative to their transfer."""
        if batch is None:
            raise ValueError("Encoder patch source does not own its CPU batch")
        start_row = segments[0].packed_offset
        source_segments = tuple(replace(segment, packed_offset=segment.packed_offset - start_row)
                                for segment in segments)
        return self.runtime.bridge.pack_payload(batch, source_segments).reshape(-1)
