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
"""Metadata, sample, and microbatch communication over injected training groups."""

from __future__ import annotations

import json
import struct
from typing import Any, Protocol, Sequence

import numpy as np

from hyper_parallel.distributed_data.schema import BatchPlan, SampleMeta, TensorShardSpec
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

platform = get_platform()


class MetadataSynchronizer(Protocol):
    """Synchronize one optimizer step's lightweight candidate metadata."""

    def gather(
        self,
        local_metadata: Sequence[SampleMeta],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[SampleMeta, ...]:
        """Gather candidates in deterministic data-owner order."""


class MicroBatchDistributor(Protocol):
    """Distribute one owner-fetched microbatch to model-parallel peers."""

    def distribute(
        self,
        micro_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return the shared plan and current rank's CP-sharded microbatch."""


class SampleRedistributor(Protocol):
    """Move owner-loaded raw samples to their planned DP data ranks."""

    def redistribute(
        self,
        local_samples: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
    ) -> dict[int, Any]:
        """Return local target samples keyed by global source position."""


class LocalMetadataSynchronizer:
    """Metadata synchronizer for a single data owner."""

    def gather(
        self,
        local_metadata: Sequence[SampleMeta],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[SampleMeta, ...]:
        """Return local candidates unchanged."""
        if len(data_owner_ranks) != 1:
            raise ValueError(f"Local metadata synchronization requires one owner, but got {data_owner_ranks}.")
        return tuple(local_metadata)


class LocalSampleRedistributor:
    """Keep online raw samples local when there is one DP data owner."""

    def redistribute(
        self,
        local_samples: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
    ) -> dict[int, Any]:
        """Return the single owner's raw samples without communication."""
        outgoing, _ = _build_sample_routes(local_samples, plan, topology)
        if len(outgoing) != 1:
            raise ValueError("Local sample redistribution requires one data rank.")
        return dict(outgoing[0])


class TorchMetadataAllGather:
    """PyTorch object all-gather for lightweight ``SampleMeta`` candidates.

    This exchanges metadata only. Pixel data and model-input tensors never
    pass through the object collective.
    """

    def __init__(self, group: Any) -> None:
        """Initialize metadata synchronization over an existing owner group."""
        if platform.platform_type != PlatformType.PYTORCH:
            raise ValueError("TorchMetadataAllGather is supported only on the PyTorch platform.")
        self._group = group

    def gather(
        self,
        local_metadata: Sequence[SampleMeta],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[SampleMeta, ...]:
        """All-gather metadata and concatenate it in ``data_owner_ranks`` order."""
        group_ranks = tuple(platform.get_process_group_ranks(self._group))
        if set(group_ranks) != set(data_owner_ranks):
            raise ValueError(f"metadata_group ranks must be {data_owner_ranks}, but got {group_ranks}.")
        gathered: list[Any] = [None] * len(group_ranks)
        platform.all_gather_object(gathered, tuple(local_metadata), self._group)
        by_global_rank = dict(zip(group_ranks, gathered, strict=True))
        ordered = []
        for data_owner_rank in data_owner_ranks:
            contribution = by_global_rank[data_owner_rank]
            if not isinstance(contribution, tuple) or any(
                not isinstance(metadata, SampleMeta) for metadata in contribution
            ):
                raise ValueError(f"Rank {data_owner_rank} contributed invalid SampleMeta data.")
            ordered.extend(contribution)
        return tuple(ordered)


class TorchPackedBytesRedistributor:
    """Exchange nested raw byte samples with variable-split tensor all-to-all."""

    def __init__(self, group: Any, *, communication_device: Any = None) -> None:
        """Initialize packed-byte A2A over the data-owner group."""
        if platform.platform_type != PlatformType.PYTORCH:
            raise ValueError("TorchPackedBytesRedistributor is supported only on the PyTorch platform.")
        self._group = group
        self._communication_device = communication_device

    def redistribute(
        self,
        local_samples: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
    ) -> dict[int, Any]:
        """Encode raw samples, exchange packed uint8 tensors, and decode local samples."""
        group_data_ranks = _data_owner_group_data_ranks(self._group, topology)
        outgoing, received_positions = _build_sample_routes(local_samples, plan, topology)
        outgoing = tuple(outgoing[data_rank] for data_rank in group_data_ranks)
        received_positions = tuple(received_positions[data_rank] for data_rank in group_data_ranks)
        segments = tuple(_pack_payload_segment(items) for items in outgoing)
        input_splits = [len(segment) for segment in segments]
        communication_device = _collective_device(self._group, self._communication_device)

        size_tensor = platform.tensor(
            input_splits,
            dtype=platform.tensor_dtype.int64,
            device=communication_device,
        )
        received_sizes_tensor, size_work = platform.all_to_all_single(
            size_tensor,
            [topology.data_parallel_size],
            self._group,
            async_op=True,
        )
        _wait_collective(size_work)
        output_splits = [int(size) for size in platform.tensor_to_numpy(received_sizes_tensor).reshape(-1)]
        if len(output_splits) != topology.data_parallel_size or any(size < 0 for size in output_splits):
            raise ValueError(f"Packed-byte A2A received invalid byte splits {output_splits}.")

        send_tensor = _bytes_to_tensor(b"".join(segments), communication_device)
        received_tensor, data_work = platform.variable_all_to_all_single(
            send_tensor,
            input_splits,
            output_splits,
            self._group,
            async_op=True,
        )
        _wait_collective(data_work)
        received_bytes = platform.tensor_to_numpy(received_tensor).tobytes()

        samples_by_position = {}
        cursor = 0
        for source_data_rank, segment_size in enumerate(output_splits):
            segment = received_bytes[cursor:cursor + segment_size]
            samples_by_position.update(_unpack_payload_segment(segment, received_positions[source_data_rank]))
            cursor += segment_size
        if cursor != len(received_bytes):
            raise ValueError("Packed-byte A2A output contains unconsumed bytes.")
        return samples_by_position


class TorchTensorRedistributor:
    """Exchange uniform tensor samples directly with variable-split all-to-all."""

    def __init__(self, group: Any, *, communication_device: Any = None) -> None:
        """Initialize direct tensor A2A over the data-owner group."""
        if platform.platform_type != PlatformType.PYTORCH:
            raise ValueError("TorchTensorRedistributor is supported only on the PyTorch platform.")
        self._group = group
        self._communication_device = communication_device

    def redistribute(
        self,
        local_samples: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
    ) -> dict[int, Any]:
        """Exchange tensor samples without Host serialization."""
        group_data_ranks = _data_owner_group_data_ranks(self._group, topology)
        outgoing, received_positions = _build_sample_routes(local_samples, plan, topology)
        outgoing = tuple(outgoing[data_rank] for data_rank in group_data_ranks)
        received_positions = tuple(received_positions[data_rank] for data_rank in group_data_ranks)
        prepared = _prepare_uniform_tensors(
            local_samples,
            self._group,
            self._communication_device,
            topology.data_parallel_size,
        )
        samples_by_position = {
            source_position: prepared[local_index]
            for local_index, source_position in enumerate(
                range(
                    topology.data_rank * len(local_samples),
                    (topology.data_rank + 1) * len(local_samples),
                )
            )
        }
        ordered_tensors = [
            samples_by_position[source_position]
            for target_items in outgoing
            for source_position, _ in target_items
        ]
        sample_shape = tuple(prepared[0].shape)
        send_tensor = platform.cat(
            [tensor.reshape((1, *sample_shape)) for tensor in ordered_tensors],
            dim=0,
        )
        input_splits = [len(items) for items in outgoing]
        output_splits = [len(positions) for positions in received_positions]
        received_tensor, work = platform.variable_all_to_all_single(
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
                f"Direct tensor A2A expected {len(ordered_positions)} samples, "
                f"but received {received_tensor.shape[0]}."
            )
        return {
            source_position: received_tensor[index]
            for index, source_position in enumerate(ordered_positions)
        }


class LocalMicroBatchDistributor:
    """Identity microbatch distribution with optional local CP slicing."""

    def distribute(
        self,
        micro_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return the owner's microbatch after applying plan-defined CP shards."""
        if micro_batch is None or plan is None:
            raise ValueError("Local distribution requires both micro_batch and plan on the data owner.")
        sharded = shard_micro_batch(micro_batch, plan.cp_shards, topology.cp_rank, topology.cp_size)
        return plan, sharded


class TorchMicroBatchDistributor:
    """Broadcast an owner microbatch over an injected model-consumer group.

    Tensor structure metadata is exchanged separately from tensor storage.
    Tensors themselves use the backend collective, so an HCCL group requires
    owner tensors and receiver allocations on their local NPU devices.
    """

    def __init__(self, group: Any, *, communication_device: Any = None) -> None:
        """Initialize microbatch broadcast over an existing consumer group."""
        if platform.platform_type != PlatformType.PYTORCH:
            raise ValueError("TorchMicroBatchDistributor is supported only on the PyTorch platform.")
        self._group = group
        self._communication_device = communication_device

    def distribute(
        self,
        micro_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Broadcast one nested microbatch, then apply this rank's CP slicing."""
        group_ranks = tuple(platform.get_process_group_ranks(self._group))
        if set(group_ranks) != set(topology.model_parallel_ranks):
            raise ValueError(
                f"model_parallel_group ranks must be {topology.model_parallel_ranks}, but got {group_ranks}."
            )
        is_source = topology.global_rank == topology.data_owner_rank
        backend = str(platform.get_backend(self._group)).lower()
        accelerator_backend = "hccl" in backend or "nccl" in backend
        if is_source and (micro_batch is None or plan is None):
            raise ValueError("The data owner must provide both micro_batch and plan.")
        if not is_source and micro_batch is not None:
            raise ValueError("Only the data owner may provide a microbatch for distribution.")
        if not is_source and accelerator_backend and self._communication_device is None:
            raise ValueError("communication_device is required to receive tensors over an HCCL/NCCL consumer group.")

        descriptor = None
        tensors = []
        if is_source:
            descriptor, tensors = _encode_payload(micro_batch)
            if accelerator_backend and any(str(tensor.device).lower().startswith("cpu") for tensor in tensors):
                raise ValueError(
                    "HCCL/NCCL microbatch tensors must be moved to the accelerator by "
                    "prepare_micro_batch before broadcast."
                )
        header = (plan, descriptor) if is_source else None
        gathered_headers: list[Any] = [None] * len(group_ranks)
        platform.all_gather_object(gathered_headers, header, self._group)
        source_index = group_ranks.index(topology.data_owner_rank)
        source_header = gathered_headers[source_index]
        if not isinstance(source_header, tuple) or len(source_header) != 2:
            raise ValueError(f"Data owner {topology.data_owner_rank} did not publish a valid microbatch header.")
        received_plan, received_descriptor = source_header
        if not isinstance(received_plan, BatchPlan):
            raise ValueError("The distributed microbatch header does not contain a valid BatchPlan.")

        if not is_source:
            micro_batch, tensors = _decode_payload(received_descriptor, self._communication_device)
        for tensor in tensors:
            platform.broadcast(tensor, topology.data_owner_rank, self._group, async_op=False)

        sharded = shard_micro_batch(
            micro_batch,
            received_plan.cp_shards,
            topology.cp_rank,
            topology.cp_size,
        )
        return received_plan, sharded


def _data_owner_group_data_ranks(group: Any, topology: DataTopology) -> tuple[int, ...]:
    group_ranks = tuple(platform.get_process_group_ranks(group))
    if set(group_ranks) != set(topology.data_owner_ranks):
        raise ValueError(f"metadata_group ranks must be {topology.data_owner_ranks}, but got {group_ranks}.")
    return tuple(topology.data_owner_ranks.index(global_rank) for global_rank in group_ranks)


def _collective_device(group: Any, communication_device: Any) -> Any:
    backend = str(platform.get_backend(group)).lower()
    if "hccl" in backend or "nccl" in backend:
        if communication_device is None:
            raise ValueError("communication_device is required for HCCL/NCCL sample A2A.")
        return communication_device
    return None


def _wait_collective(work: Any) -> None:
    if work is not None:
        work.wait()


def _build_sample_routes(
    local_samples: Sequence[Any],
    plan: BatchPlan,
    topology: DataTopology,
) -> tuple[tuple[tuple[tuple[int, Any], ...], ...], tuple[tuple[int, ...], ...]]:
    if not topology.is_data_owner:
        raise ValueError("Only data owners may redistribute online raw samples.")
    if plan.data_parallel_size != topology.data_parallel_size:
        raise ValueError(
            f"Plan data_parallel_size={plan.data_parallel_size} does not match topology "
            f"data_parallel_size={topology.data_parallel_size}."
        )
    local_count = plan.micro_batch_size * plan.micro_batch_num
    if len(local_samples) != local_count:
        raise ValueError(f"Expected {local_count} local raw samples, but got {len(local_samples)}.")
    planned_by_position = {sample.source_position: sample for sample in plan.samples}
    if len(planned_by_position) != len(plan.samples):
        raise ValueError("BatchPlan source positions must be unique for raw sample redistribution.")

    source_start = topology.data_rank * local_count
    outgoing: list[list[tuple[int, Any]]] = [[] for _ in range(topology.data_parallel_size)]
    for local_index, sample in enumerate(local_samples):
        source_position = source_start + local_index
        planned_sample = planned_by_position.get(source_position)
        if planned_sample is None:
            raise ValueError(f"BatchPlan is missing source position {source_position}.")
        outgoing[planned_sample.target_data_rank].append((source_position, sample))

    received_positions: list[list[int]] = [[] for _ in range(topology.data_parallel_size)]
    for planned_sample in plan.samples:
        if planned_sample.target_data_rank != topology.data_rank:
            continue
        source_data_rank = planned_sample.source_position // local_count
        received_positions[source_data_rank].append(planned_sample.source_position)
    for positions in received_positions:
        positions.sort()
    return (
        tuple(tuple(items) for items in outgoing),
        tuple(tuple(positions) for positions in received_positions),
    )


def _prepare_uniform_tensors(
    local_samples: Sequence[Any],
    group: Any,
    communication_device: Any,
    data_parallel_size: int,
) -> tuple[Any, ...]:
    if not local_samples or any(not platform.is_tensor(sample) for sample in local_samples):
        raise ValueError("direct_tensor_a2a requires every online sample to be a tensor.")
    sample_shape = tuple(local_samples[0].shape)
    sample_dtype = local_samples[0].dtype
    if any(tuple(sample.shape) != sample_shape or sample.dtype != sample_dtype for sample in local_samples):
        raise ValueError("direct_tensor_a2a requires identical sample shapes and dtypes on each data owner.")

    local_schema = (sample_shape, str(sample_dtype))
    gathered_schemas: list[Any] = [None] * data_parallel_size
    platform.all_gather_object(gathered_schemas, local_schema, group)
    if any(schema != local_schema for schema in gathered_schemas):
        raise ValueError(f"direct_tensor_a2a requires one global tensor schema, but got {gathered_schemas}.")

    backend = str(platform.get_backend(group)).lower()
    accelerator_backend = "hccl" in backend or "nccl" in backend
    prepared = []
    for sample in local_samples:
        tensor = sample.contiguous()
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
    tensor = platform.from_numpy(host_array)
    if communication_device is not None:
        tensor = tensor.to(communication_device, non_blocking=True)
    return tensor


def _encode_binary_payload(payload: Any) -> bytes:
    blobs = []

    def _describe(value: Any) -> Any:
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
        if value is None or isinstance(value, (bool, int, float, str)):
            return ["value", value]
        raise ValueError(
            "packed_bytes_a2a supports nested bytes, dict, list, tuple, and JSON scalar values, "
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


def shard_micro_batch(
    micro_batch: Any,
    shard_specs: tuple[TensorShardSpec, ...],
    cp_rank: int,
    cp_size: int,
) -> Any:
    """Shard selected nested tensor fields for one CP rank.

    Args:
        micro_batch: Nested batch structure.
        shard_specs: Plan-defined tensor paths and shard dimensions.
        cp_rank: Local rank in the CP group.
        cp_size: CP group size.

    Returns:
        Microbatch with selected tensor leaves replaced by local CP shards.
    """
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), but got {cp_rank}.")
    if cp_size == 1 or not shard_specs:
        return micro_batch
    spec_by_path = {spec.path: spec for spec in shard_specs}
    if len(spec_by_path) != len(shard_specs):
        raise ValueError("BatchPlan.cp_shards must not contain duplicate tensor paths.")
    seen_paths: set[tuple[str | int, ...]] = set()

    def _apply(value: Any, path: tuple[str | int, ...]) -> Any:
        spec = spec_by_path.get(path)
        if spec is not None:
            if not platform.is_tensor(value):
                raise ValueError(f"CP shard path {path} does not address a tensor microbatch leaf.")
            dimension = spec.dim + len(value.shape) if spec.dim < 0 else spec.dim
            if dimension < 0 or dimension >= len(value.shape):
                raise ValueError(f"CP shard dim {spec.dim} is invalid for tensor at {path} with shape {value.shape}.")
            if value.shape[dimension] % cp_size != 0:
                raise ValueError(
                    f"Tensor at {path} has size {value.shape[dimension]} on dim {spec.dim}, "
                    f"which is not divisible by cp_size={cp_size}."
                )
            seen_paths.add(path)
            return platform.chunk(value, spec.dim, cp_size, cp_rank)
        if isinstance(value, dict):
            return {key: _apply(child, path + (key,)) for key, child in value.items()}
        if isinstance(value, list):
            return [_apply(child, path + (index,)) for index, child in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(_apply(child, path + (index,)) for index, child in enumerate(value))
        return value

    result = _apply(micro_batch, ())
    missing_paths = set(spec_by_path) - seen_paths
    if missing_paths:
        raise ValueError(f"CP shard paths were not found in microbatch: {sorted(missing_paths, key=repr)}.")
    return result


def _encode_payload(payload: Any) -> tuple[Any, list[Any]]:
    tensors = []

    def _encode(value: Any) -> Any:
        if platform.is_tensor(value):
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
            tensor = platform.new_tensor(node[1], node[2], communication_device)
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
