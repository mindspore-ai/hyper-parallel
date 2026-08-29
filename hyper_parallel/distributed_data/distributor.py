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
"""Metadata and local-batch communication over injected training groups."""
# This package is intentionally PyTorch-only.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import json
import math
import struct
from typing import Any, Protocol, Sequence, TypeVar

import numpy as np
import torch
import torch.distributed as dist

from hyper_parallel.distributed_data.schema import (
    BatchPlan,
    LocalBatchMeta,
    OnlineLocalBatchMetadata,
    TensorLocalBatchSpec,
    TensorShardSpec,
)
from hyper_parallel.distributed_data.topology import DataTopology

MetadataValue = TypeVar("MetadataValue", LocalBatchMeta, OnlineLocalBatchMetadata)


class MetadataSynchronizer(Protocol):
    """Synchronize one optimizer step's lightweight candidate metadata."""

    def gather(
        self,
        local_metadata: Sequence[MetadataValue],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[MetadataValue, ...]:
        """Gather candidates in deterministic data-owner order."""


class ModelParallelLocalBatchDistributor(Protocol):
    """Distribute one owner local batch to model-parallel peers."""

    def distribute(
        self,
        local_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return the shared plan and current rank's CP-sharded local batch."""


class LocalBatchRedistributor(Protocol):
    """Move online local batches to their planned DP data ranks."""

    def describe_local_batch(self, local_batch: Any) -> TensorLocalBatchSpec | None:
        """Return metadata needed to transport one online local batch."""

    def redistribute(
        self,
        local_batches: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
        global_metadata: Sequence[OnlineLocalBatchMetadata],
    ) -> dict[int, Any]:
        """Return target local batches keyed by global source position."""


class LocalMetadataSynchronizer:
    """Metadata synchronizer for a single data owner."""

    def gather(
        self,
        local_metadata: Sequence[MetadataValue],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[MetadataValue, ...]:
        """Return local candidates unchanged."""
        if len(data_owner_ranks) != 1:
            raise ValueError(f"Local metadata synchronization requires one owner, but got {data_owner_ranks}.")
        return tuple(local_metadata)


class IdentityLocalBatchRedistributor:
    """Keep online local batches in place when there is one DP data owner."""

    @staticmethod
    def describe_local_batch(local_batch: Any) -> TensorLocalBatchSpec | None:
        """Return no descriptor because the local batch does not communicate."""
        del local_batch

    def redistribute(
        self,
        local_batches: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
        global_metadata: Sequence[OnlineLocalBatchMetadata],
    ) -> dict[int, Any]:
        """Return the single owner's local batches without communication."""
        del global_metadata
        outgoing, _ = _build_local_batch_routes(local_batches, plan, topology)
        if len(outgoing) != 1:
            raise ValueError("Local-batch redistribution requires one data rank.")
        return dict(outgoing[0])


class TorchMetadataAllGather:
    """PyTorch object all-gather for lightweight local-batch metadata.

    Online tensor shape descriptors share this collective with planner
    metadata. Pixel data and model-input tensor storage never pass through it.
    """

    def __init__(self, group: Any) -> None:
        """Initialize metadata synchronization over an existing owner group."""
        self._group = group

    def gather(
        self,
        local_metadata: Sequence[MetadataValue],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[MetadataValue, ...]:
        """All-gather metadata and concatenate it in ``data_owner_ranks`` order."""
        group_ranks = tuple(dist.get_process_group_ranks(self._group))
        if set(group_ranks) != set(data_owner_ranks):
            raise ValueError(f"metadata_group ranks must be {data_owner_ranks}, but got {group_ranks}.")
        gathered: list[Any] = [None] * len(group_ranks)
        dist.all_gather_object(gathered, tuple(local_metadata), self._group)
        by_global_rank = dict(zip(group_ranks, gathered, strict=True))
        ordered = []
        for data_owner_rank in data_owner_ranks:
            contribution = by_global_rank[data_owner_rank]
            if not isinstance(contribution, tuple) or any(
                not isinstance(metadata, (LocalBatchMeta, OnlineLocalBatchMetadata))
                for metadata in contribution
            ):
                raise ValueError(f"Rank {data_owner_rank} contributed invalid local-batch metadata.")
            ordered.extend(contribution)
        return tuple(ordered)


class TorchPackedBytesLocalBatchRedistributor:
    """Exchange nested local batches with variable-split tensor all-to-all."""

    def __init__(self, group: Any, *, communication_device: Any = None) -> None:
        """Initialize packed-byte A2A over the data-owner group."""
        self._group = group
        self._communication_device = communication_device

    @staticmethod
    def describe_local_batch(local_batch: Any) -> TensorLocalBatchSpec | None:
        """Return no tensor descriptor for packed byte transport."""
        del local_batch

    def redistribute(
        self,
        local_batches: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
        global_metadata: Sequence[OnlineLocalBatchMetadata],
    ) -> dict[int, Any]:
        """Encode local batches, exchange packed uint8 tensors, and decode received batches."""
        del global_metadata
        group_data_ranks = _data_owner_group_data_ranks(self._group, topology)
        outgoing, received_positions = _build_local_batch_routes(local_batches, plan, topology)
        outgoing = tuple(outgoing[data_rank] for data_rank in group_data_ranks)
        received_positions = tuple(received_positions[data_rank] for data_rank in group_data_ranks)
        segments = tuple(_pack_payload_segment(items) for items in outgoing)
        input_splits = [len(segment) for segment in segments]
        communication_device = _collective_device(self._group, self._communication_device)

        size_tensor = torch.tensor(
            input_splits,
            dtype=torch.int64,
            device=communication_device,
        )
        received_sizes_tensor, size_work = _all_to_all_single(
            size_tensor,
            [topology.data_parallel_size],
            self._group,
            async_op=True,
        )
        _wait_collective(size_work)
        output_splits = [int(size) for size in _tensor_to_numpy(received_sizes_tensor).reshape(-1)]
        if len(output_splits) != topology.data_parallel_size or any(size < 0 for size in output_splits):
            raise ValueError(f"Packed-byte A2A received invalid byte splits {output_splits}.")

        send_tensor = _bytes_to_tensor(b"".join(segments), communication_device)
        received_tensor, data_work = _variable_all_to_all_single(
            send_tensor,
            input_splits,
            output_splits,
            self._group,
            async_op=True,
        )
        _wait_collective(data_work)
        received_bytes = _tensor_to_numpy(received_tensor).tobytes()

        local_batches_by_position = {}
        cursor = 0
        for source_data_rank, segment_size in enumerate(output_splits):
            segment = received_bytes[cursor:cursor + segment_size]
            local_batches_by_position.update(
                _unpack_payload_segment(segment, received_positions[source_data_rank])
            )
            cursor += segment_size
        if cursor != len(received_bytes):
            raise ValueError("Packed-byte A2A output contains unconsumed bytes.")
        return local_batches_by_position


class TorchTensorLocalBatchRedistributor:
    """Exchange fixed- or variable-shape tensor local batches with all-to-all."""

    def __init__(self, group: Any, *, communication_device: Any = None) -> None:
        """Initialize direct tensor A2A over the data-owner group."""
        self._group = group
        self._communication_device = communication_device

    @staticmethod
    def describe_local_batch(local_batch: Any) -> TensorLocalBatchSpec | None:
        """Build the tensor descriptor synchronized by the existing metadata All-Gather."""
        if not torch.is_tensor(local_batch):
            return None
        shape = tuple(int(size) for size in local_batch.shape)
        return TensorLocalBatchSpec(shape=shape, dtype=str(local_batch.dtype), numel=math.prod(shape))

    def redistribute(
        self,
        local_batches: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
        global_metadata: Sequence[OnlineLocalBatchMetadata],
    ) -> dict[int, Any]:
        """Select a uniform fast path or flattened variable-shape A2A from global metadata."""
        group_data_ranks = _data_owner_group_data_ranks(self._group, topology)
        outgoing, received_positions = _build_local_batch_routes(local_batches, plan, topology)
        outgoing = tuple(outgoing[data_rank] for data_rank in group_data_ranks)
        received_positions = tuple(received_positions[data_rank] for data_rank in group_data_ranks)
        tensor_specs = _validate_tensor_metadata(global_metadata, plan)
        prepared = _prepare_tensors(
            local_batches,
            tensor_specs,
            topology.data_rank,
            self._group,
            self._communication_device,
        )
        local_batches_by_position = {
            source_position: prepared[local_index]
            for local_index, source_position in enumerate(
                range(
                    topology.data_rank * len(local_batches),
                    (topology.data_rank + 1) * len(local_batches),
                )
            )
        }
        ordered_tensors = [
            local_batches_by_position[source_position]
            for target_items in outgoing
            for source_position, _ in target_items
        ]
        if len({spec.shape for spec in tensor_specs}) == 1:
            return self._redistribute_uniform(
                ordered_tensors,
                outgoing,
                received_positions,
                tensor_specs,
            )
        return self._redistribute_variable(
            ordered_tensors,
            outgoing,
            received_positions,
            tensor_specs,
        )

    def _redistribute_uniform(
        self,
        ordered_tensors: Sequence[Any],
        outgoing: Sequence[Sequence[tuple[int, Any]]],
        received_positions: Sequence[Sequence[int]],
        tensor_specs: Sequence[TensorLocalBatchSpec],
    ) -> dict[int, Any]:
        local_batch_shape = tensor_specs[0].shape
        send_tensor = torch.cat(
            [tensor.reshape((1, *local_batch_shape)) for tensor in ordered_tensors],
            dim=0,
        )
        input_splits = [len(items) for items in outgoing]
        output_splits = [len(positions) for positions in received_positions]
        received_tensor, work = _variable_all_to_all_single(
            send_tensor,
            input_splits,
            output_splits,
            self._group,
            async_op=True,
        )
        _wait_collective(work)
        ordered_positions = [position for positions in received_positions for position in positions]
        if received_tensor.shape[0] != len(ordered_positions):
            raise ValueError(
                f"Direct tensor A2A expected {len(ordered_positions)} local batches, "
                f"but received {received_tensor.shape[0]}."
            )
        return {
            source_position: received_tensor[index]
            for index, source_position in enumerate(ordered_positions)
        }

    def _redistribute_variable(
        self,
        ordered_tensors: Sequence[Any],
        outgoing: Sequence[Sequence[tuple[int, Any]]],
        received_positions: Sequence[Sequence[int]],
        tensor_specs: Sequence[TensorLocalBatchSpec],
    ) -> dict[int, Any]:
        send_tensor = torch.cat([tensor.reshape((-1,)) for tensor in ordered_tensors], dim=0)
        input_splits = [
            sum(tensor_specs[source_position].numel for source_position, _ in items)
            for items in outgoing
        ]
        output_splits = [
            sum(tensor_specs[source_position].numel for source_position in positions)
            for positions in received_positions
        ]
        received_tensor, work = _variable_all_to_all_single(
            send_tensor,
            input_splits,
            output_splits,
            self._group,
            async_op=True,
        )
        _wait_collective(work)
        local_batches_by_position = {}
        cursor = 0
        for source_positions in received_positions:
            for source_position in source_positions:
                tensor_spec = tensor_specs[source_position]
                end = cursor + tensor_spec.numel
                local_batches_by_position[source_position] = received_tensor[cursor:end].reshape(tensor_spec.shape)
                cursor = end
        if cursor != received_tensor.shape[0]:
            raise ValueError(
                f"Variable tensor A2A consumed {cursor} elements, but received {received_tensor.shape[0]}."
            )
        return local_batches_by_position


class IdentityModelParallelLocalBatchDistributor:
    """Identity local-batch distribution with optional local CP slicing."""

    def distribute(
        self,
        local_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return the owner's local batch after applying plan-defined CP shards."""
        if local_batch is None or plan is None:
            raise ValueError("Local distribution requires both local_batch and plan on the data owner.")
        sharded = shard_local_batch(local_batch, plan.cp_shards, topology.cp_rank, topology.cp_size)
        return plan, sharded


class TorchModelParallelLocalBatchDistributor:
    """Broadcast an owner local batch over an injected model-consumer group.

    Tensor structure metadata is exchanged separately from tensor storage.
    Tensors themselves use the backend collective, so an HCCL group requires
    owner tensors and receiver allocations on their local NPU devices.
    """

    def __init__(self, group: Any, *, communication_device: Any = None) -> None:
        """Initialize local-batch broadcast over an existing consumer group."""
        self._group = group
        self._communication_device = communication_device

    def distribute(
        self,
        local_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Broadcast one nested local batch, then apply this rank's CP slicing."""
        group_ranks = tuple(dist.get_process_group_ranks(self._group))
        if set(group_ranks) != set(topology.model_parallel_ranks):
            raise ValueError(
                f"model_parallel_group ranks must be {topology.model_parallel_ranks}, but got {group_ranks}."
            )
        is_source = topology.global_rank == topology.data_owner_rank
        backend = str(dist.get_backend(self._group)).lower()
        accelerator_backend = "hccl" in backend or "nccl" in backend
        if is_source and (local_batch is None or plan is None):
            raise ValueError("The data owner must provide both local_batch and plan.")
        if not is_source and local_batch is not None:
            raise ValueError("Only the data owner may provide a local batch for distribution.")
        if not is_source and accelerator_backend and self._communication_device is None:
            raise ValueError("communication_device is required to receive tensors over an HCCL/NCCL consumer group.")

        descriptor = None
        tensors = []
        if is_source:
            descriptor, tensors = _encode_payload(local_batch)
            if accelerator_backend and any(str(tensor.device).lower().startswith("cpu") for tensor in tensors):
                raise ValueError(
                    "HCCL/NCCL local-batch tensors must be moved to the accelerator by "
                    "prepare_local_batch before broadcast."
                )
        header = (plan, descriptor) if is_source else None
        gathered_headers: list[Any] = [None] * len(group_ranks)
        dist.all_gather_object(gathered_headers, header, self._group)
        source_index = group_ranks.index(topology.data_owner_rank)
        source_header = gathered_headers[source_index]
        if not isinstance(source_header, tuple) or len(source_header) != 2:
            raise ValueError(f"Data owner {topology.data_owner_rank} did not publish a valid local-batch header.")
        received_plan, received_descriptor = source_header
        if not isinstance(received_plan, BatchPlan):
            raise ValueError("The distributed local-batch header does not contain a valid BatchPlan.")

        if not is_source:
            local_batch, tensors = _decode_payload(received_descriptor, self._communication_device)
        for tensor in tensors:
            dist.broadcast(tensor, src=topology.data_owner_rank, group=self._group, async_op=False)

        sharded = shard_local_batch(
            local_batch,
            received_plan.cp_shards,
            topology.cp_rank,
            topology.cp_size,
        )
        return received_plan, sharded


def _data_owner_group_data_ranks(group: Any, topology: DataTopology) -> tuple[int, ...]:
    group_ranks = tuple(dist.get_process_group_ranks(group))
    if set(group_ranks) != set(topology.data_owner_ranks):
        raise ValueError(f"metadata_group ranks must be {topology.data_owner_ranks}, but got {group_ranks}.")
    return tuple(topology.data_owner_ranks.index(global_rank) for global_rank in group_ranks)


def _collective_device(group: Any, communication_device: Any) -> Any:
    backend = str(dist.get_backend(group)).lower()
    if "hccl" in backend or "nccl" in backend:
        if communication_device is None:
            raise ValueError("communication_device is required for HCCL/NCCL local-batch A2A.")
        return communication_device
    return None


def _all_to_all_single(
    input_tensor: torch.Tensor,
    output_shape: Sequence[int],
    group: Any,
    *,
    async_op: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Allocate output and run a fixed-shape PyTorch all-to-all."""
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    work = dist.all_to_all_single(output, input_tensor, group=group, async_op=async_op)
    return output, work


def _variable_all_to_all_single(
    input_tensor: torch.Tensor,
    input_splits: Sequence[int],
    output_splits: Sequence[int],
    group: Any,
    *,
    async_op: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Allocate output and run a variable-split PyTorch all-to-all."""
    output = torch.empty(
        (sum(output_splits), *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    work = dist.all_to_all_single(
        output,
        input_tensor,
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
        async_op=async_op,
    )
    return output, work


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Copy a PyTorch tensor to Host NumPy storage."""
    return tensor.cpu().numpy()


def _str_to_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a serialized PyTorch dtype name."""
    prefix, separator, name = dtype_name.partition(".")
    if prefix != "torch" or separator != "." or not name:
        raise ValueError(f"Expected dtype string like 'torch.float32', got {dtype_name!r}.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported PyTorch dtype {dtype_name!r}.")
    return dtype


def _wait_collective(work: Any) -> None:
    if work is not None:
        work.wait()


def _build_local_batch_routes(
    local_batches: Sequence[Any],
    plan: BatchPlan,
    topology: DataTopology,
) -> tuple[tuple[tuple[tuple[int, Any], ...], ...], tuple[tuple[int, ...], ...]]:
    if not topology.is_data_owner:
        raise ValueError("Only data owners may redistribute online local batches.")
    if plan.data_parallel_size != topology.data_parallel_size:
        raise ValueError(
            f"Plan data_parallel_size={plan.data_parallel_size} does not match topology "
            f"data_parallel_size={topology.data_parallel_size}."
        )
    local_count = plan.micro_batch_num
    if len(local_batches) != local_count:
        raise ValueError(f"Expected {local_count} local batches, but got {len(local_batches)}.")
    planned_by_position = {
        local_batch.source_position: local_batch
        for local_batch in plan.local_batches
    }
    if len(planned_by_position) != len(plan.local_batches):
        raise ValueError("BatchPlan source positions must be unique for local-batch redistribution.")

    source_start = topology.data_rank * local_count
    outgoing: list[list[tuple[int, Any]]] = [[] for _ in range(topology.data_parallel_size)]
    for local_index, local_batch in enumerate(local_batches):
        source_position = source_start + local_index
        planned_local_batch = planned_by_position.get(source_position)
        if planned_local_batch is None:
            raise ValueError(f"BatchPlan is missing source position {source_position}.")
        outgoing[planned_local_batch.target_data_rank].append((source_position, local_batch))

    received_positions: list[list[int]] = [[] for _ in range(topology.data_parallel_size)]
    for planned_local_batch in plan.local_batches:
        if planned_local_batch.target_data_rank != topology.data_rank:
            continue
        source_data_rank = planned_local_batch.source_position // local_count
        received_positions[source_data_rank].append(planned_local_batch.source_position)
    for positions in received_positions:
        positions.sort()
    return (
        tuple(tuple(items) for items in outgoing),
        tuple(tuple(positions) for positions in received_positions),
    )


def _validate_tensor_metadata(
    global_metadata: Sequence[OnlineLocalBatchMetadata],
    plan: BatchPlan,
) -> tuple[TensorLocalBatchSpec, ...]:
    if len(global_metadata) != len(plan.local_batches):
        raise ValueError(
            f"Direct tensor A2A expected {len(plan.local_batches)} global metadata entries, "
            f"but got {len(global_metadata)}."
        )
    planned_by_position = {
        local_batch.source_position: local_batch
        for local_batch in plan.local_batches
    }
    tensor_specs = []
    for source_position, metadata in enumerate(global_metadata):
        planned_local_batch = planned_by_position.get(source_position)
        if (
            planned_local_batch is None
            or planned_local_batch.meta.local_batch_id != metadata.local_batch_meta.local_batch_id
        ):
            raise ValueError(f"Tensor metadata at source position {source_position} does not match BatchPlan.")
        if metadata.tensor_spec is None:
            raise ValueError(f"Direct tensor A2A is missing tensor metadata at source position {source_position}.")
        tensor_specs.append(metadata.tensor_spec)
    dtypes = {spec.dtype for spec in tensor_specs}
    if len(dtypes) != 1:
        raise ValueError(f"direct_tensor_a2a requires one dtype per global microbatch, but got {sorted(dtypes)}.")
    return tuple(tensor_specs)


def _prepare_tensors(
    local_batches: Sequence[Any],
    tensor_specs: Sequence[TensorLocalBatchSpec],
    data_rank: int,
    group: Any,
    communication_device: Any,
) -> tuple[Any, ...]:
    if not local_batches or any(not torch.is_tensor(local_batch) for local_batch in local_batches):
        raise ValueError("direct_tensor_a2a requires every online local batch to be a tensor.")
    local_count = len(local_batches)
    source_start = data_rank * local_count
    for local_index, local_batch in enumerate(local_batches):
        expected_spec = tensor_specs[source_start + local_index]
        local_batch_shape = tuple(int(size) for size in local_batch.shape)
        actual_spec = TensorLocalBatchSpec(
            shape=local_batch_shape,
            dtype=str(local_batch.dtype),
            numel=math.prod(local_batch_shape),
        )
        if actual_spec != expected_spec:
            raise ValueError(
                f"Tensor local batch at source position {source_start + local_index} changed after metadata sync: "
                f"expected {expected_spec}, got {actual_spec}."
            )

    backend = str(dist.get_backend(group)).lower()
    accelerator_backend = "hccl" in backend or "nccl" in backend
    prepared = []
    for local_batch in local_batches:
        tensor = local_batch.contiguous()
        if accelerator_backend and communication_device is not None:
            tensor = tensor.to(communication_device, non_blocking=True)
        elif accelerator_backend and str(tensor.device).lower().startswith("cpu"):
            raise ValueError("communication_device is required for CPU tensors over HCCL/NCCL direct A2A.")
        elif not accelerator_backend and not str(tensor.device).lower().startswith("cpu"):
            raise ValueError("CPU owner groups require CPU tensors for direct_tensor_a2a.")
        prepared.append(tensor)
    return tuple(prepared)


def _bytes_to_tensor(data: bytes, communication_device: Any) -> Any:
    host_array = np.frombuffer(bytearray(data), dtype=np.uint8)
    tensor = torch.from_numpy(host_array)
    if communication_device is not None:
        tensor = tensor.to(communication_device, non_blocking=True)
    return tensor


def _encode_binary_payload(payload: Any) -> bytes:
    blobs = []

    def _describe(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            byte_tensor = tensor.reshape((-1,)).view(torch.uint8)
            blob = _tensor_to_numpy(byte_tensor).tobytes()
            blobs.append(blob)
            return ["tensor", [list(tensor.shape), str(tensor.dtype), len(blob)]]
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            blob = array.tobytes()
            blobs.append(blob)
            return ["ndarray", [list(array.shape), array.dtype.str, len(blob)]]
        if isinstance(value, (bytes, bytearray, memoryview)):
            blob = bytes(value)
            blobs.append(blob)
            return ["bytes", len(blob)]
        if isinstance(value, dict):
            return ["dict", [[_describe(key), _describe(child)] for key, child in value.items()]]
        if isinstance(value, list):
            return ["list", [_describe(child) for child in value]]
        if isinstance(value, tuple):
            return ["tuple", [_describe(child) for child in value]]
        if isinstance(value, np.generic):
            return ["value", value.item()]
        if value is None or isinstance(value, (bool, int, float, str)):
            return ["value", value]
        raise ValueError(
            "packed_bytes_a2a supports nested tensors, arrays, bytes, dict, list, tuple, and JSON scalar values, "
            f"but got {type(value)}."
        )

    descriptor = _describe(payload)
    header = json.dumps(descriptor, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"".join(blobs)


def _decode_binary_payload(frame: bytes) -> Any:
    if len(frame) < 8:
        raise ValueError("Packed-byte payload frame is shorter than its header prefix.")
    header_size = struct.unpack_from("<Q", frame, 0)[0]
    payload_start = 8 + header_size
    if payload_start > len(frame):
        raise ValueError("Packed-byte payload frame contains an invalid header size.")
    try:
        descriptor = json.loads(frame[8:payload_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Packed-byte payload frame contains an invalid JSON descriptor.") from exc
    cursor = payload_start

    def _decode(node: Any) -> Any:
        nonlocal cursor
        if not isinstance(node, list) or len(node) != 2 or not isinstance(node[0], str):
            raise ValueError("Packed-byte payload descriptor contains an invalid node.")
        kind, value = node
        if kind == "tensor" and _valid_binary_array_descriptor(value):
            shape, dtype_name, size = value
            end = cursor + size
            if end > len(frame):
                raise ValueError("Packed-byte tensor descriptor exceeds the frame size.")
            byte_array = np.frombuffer(frame[cursor:end], dtype=np.uint8).copy()
            cursor = end
            try:
                tensor = torch.from_numpy(byte_array).view(_str_to_dtype(dtype_name))
                return tensor.reshape(tuple(shape))
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Packed-byte tensor descriptor has invalid shape {shape} or dtype {dtype_name!r}."
                ) from exc
        if kind == "ndarray" and _valid_binary_array_descriptor(value):
            shape, dtype_name, size = value
            end = cursor + size
            if end > len(frame):
                raise ValueError("Packed-byte array descriptor exceeds the frame size.")
            try:
                array = np.frombuffer(frame[cursor:end], dtype=np.dtype(dtype_name)).copy().reshape(tuple(shape))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Packed-byte array descriptor has invalid shape {shape} or dtype {dtype_name!r}."
                ) from exc
            cursor = end
            return array
        if kind == "bytes" and isinstance(value, int) and value >= 0:
            end = cursor + value
            if end > len(frame):
                raise ValueError("Packed-byte payload descriptor exceeds the frame size.")
            result = frame[cursor:end]
            cursor = end
            return result
        if kind == "dict" and isinstance(value, list):
            return {_decode(item[0]): _decode(item[1]) for item in value}
        if kind == "list" and isinstance(value, list):
            return [_decode(child) for child in value]
        if kind == "tuple" and isinstance(value, list):
            return tuple(_decode(child) for child in value)
        if kind == "value" and (value is None or isinstance(value, (bool, int, float, str))):
            return value
        raise ValueError(f"Packed-byte payload descriptor contains invalid {kind!r} data.")

    payload = _decode(descriptor)
    if cursor != len(frame):
        raise ValueError("Packed-byte payload frame contains trailing bytes.")
    return payload


def _valid_binary_array_descriptor(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    shape, dtype_name, size = value
    return (
        isinstance(shape, list)
        and all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape)
        and isinstance(dtype_name, str)
        and bool(dtype_name)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
    )


def _pack_payload_segment(items: Sequence[tuple[int, Any]]) -> bytes:
    records = []
    for _, payload in items:
        frame = _encode_binary_payload(payload)
        records.append(struct.pack("<Q", len(frame)) + frame)
    return b"".join(records)


def _unpack_payload_segment(segment: bytes, source_positions: Sequence[int]) -> dict[int, Any]:
    payload_by_position = {}
    cursor = 0
    for source_position in source_positions:
        if cursor + 8 > len(segment):
            raise ValueError("Packed-byte A2A segment is missing a record-size prefix.")
        frame_size = struct.unpack_from("<Q", segment, cursor)[0]
        frame_start = cursor + 8
        frame_end = frame_start + frame_size
        if frame_end > len(segment):
            raise ValueError("Packed-byte A2A segment contains a truncated record.")
        payload_by_position[source_position] = _decode_binary_payload(segment[frame_start:frame_end])
        cursor = frame_end
    if cursor != len(segment):
        raise ValueError("Packed-byte A2A segment contains unexpected records or trailing bytes.")
    return payload_by_position


def shard_local_batch(
    local_batch: Any,
    shard_specs: tuple[TensorShardSpec, ...],
    cp_rank: int,
    cp_size: int,
) -> Any:
    """Shard selected nested tensor fields for one CP rank.

    Args:
        local_batch: Nested local-batch structure.
        shard_specs: Plan-defined tensor paths and shard dimensions.
        cp_rank: Local rank in the CP group.
        cp_size: CP group size.

    Returns:
        Local batch with selected tensor leaves replaced by local CP shards.
    """
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), but got {cp_rank}.")
    if cp_size == 1 or not shard_specs:
        return local_batch
    spec_by_path = {spec.path: spec for spec in shard_specs}
    if len(spec_by_path) != len(shard_specs):
        raise ValueError("BatchPlan.cp_shards must not contain duplicate tensor paths.")
    seen_paths: set[tuple[str | int, ...]] = set()

    def _apply(value: Any, path: tuple[str | int, ...]) -> Any:
        spec = spec_by_path.get(path)
        if spec is not None:
            if not torch.is_tensor(value):
                raise ValueError(f"CP shard path {path} does not address a tensor local-batch leaf.")
            dimension = spec.dim + len(value.shape) if spec.dim < 0 else spec.dim
            if dimension < 0 or dimension >= len(value.shape):
                raise ValueError(f"CP shard dim {spec.dim} is invalid for tensor at {path} with shape {value.shape}.")
            if value.shape[dimension] % cp_size != 0:
                raise ValueError(
                    f"Tensor at {path} has size {value.shape[dimension]} on dim {spec.dim}, "
                    f"which is not divisible by cp_size={cp_size}."
                )
            seen_paths.add(path)
            return torch.chunk(value, cp_size, dim=spec.dim)[cp_rank]
        if isinstance(value, dict):
            return {key: _apply(child, path + (key,)) for key, child in value.items()}
        if isinstance(value, list):
            return [_apply(child, path + (index,)) for index, child in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(_apply(child, path + (index,)) for index, child in enumerate(value))
        return value

    result = _apply(local_batch, ())
    missing_paths = set(spec_by_path) - seen_paths
    if missing_paths:
        raise ValueError(f"CP shard paths were not found in local batch: {sorted(missing_paths, key=repr)}.")
    return result


def _encode_payload(payload: Any) -> tuple[Any, list[Any]]:
    tensors = []

    def _encode(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.contiguous()
            tensors.append(tensor)
            return ("tensor", tuple(tensor.shape), tensor.dtype)
        if isinstance(value, dict):
            return ("dict", tuple((key, _encode(child)) for key, child in value.items()))
        if isinstance(value, list):
            return ("list", tuple(_encode(child) for child in value))
        if isinstance(value, tuple):
            return ("tuple", tuple(_encode(child) for child in value))
        return ("value", value)

    return _encode(payload), tensors


def _decode_payload(descriptor: Any, communication_device: Any) -> tuple[Any, list[Any]]:
    tensors = []

    def _decode(node: Any) -> Any:
        if not isinstance(node, tuple) or not node:
            raise ValueError(f"Invalid payload descriptor node: {node!r}.")
        kind = node[0]
        if kind == "tensor" and len(node) == 3:
            tensor = torch.empty(size=node[1], dtype=node[2], device=communication_device)
            tensors.append(tensor)
            return tensor
        if kind == "dict" and len(node) == 2:
            return {key: _decode(child) for key, child in node[1]}
        if kind == "list" and len(node) == 2:
            return [_decode(child) for child in node[1]]
        if kind == "tuple" and len(node) == 2:
            return tuple(_decode(child) for child in node[1])
        if kind == "value" and len(node) == 2:
            return node[1]
        raise ValueError(f"Invalid payload descriptor node: {node!r}.")

    return _decode(descriptor), tensors
