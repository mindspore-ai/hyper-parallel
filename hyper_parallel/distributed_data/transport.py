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
"""CPU control, direct tensor broadcast, and configurable payload data planes."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.topology import DataTopology

@dataclass(frozen=True)
class DataGroups:
    """Control, payload, and model-consumer groups for distributed data."""

    data_plane_ranks: tuple[int, ...]
    control_group: Any
    payload_group: Any
    model_parallel_group: Any
    planner_rank: int
    distributed: bool
    model_parallel_tensor_group: Any = None


@dataclass(frozen=True)
class PreparedPayloadExchange:
    """Serialized host payload and collective tensor for one payload A2A."""

    input_splits: tuple[int, ...]
    send_storage: bytearray
    send_tensor: torch.Tensor
    local_segment: bytes


def _is_accelerator_backend(backend: str) -> bool:
    return "hccl" in backend.lower() or "nccl" in backend.lower()


def _control_backend(group: Any, backend: str | None = None) -> str:
    if backend is not None:
        return backend.lower()
    try:
        return str(dist.get_backend(group)).lower()
    except (RuntimeError, ValueError):
        # Lightweight unit-test groups and single-process fakes represent Gloo
        # groups with opaque sentinels rather than registered ProcessGroups.
        return "gloo"


def _require_control_device(device: Any, backend: str) -> torch.device:
    if device is None:
        raise ValueError(f"{backend} control collectives require an accelerator communication_device.")
    resolved = torch.device(device)
    expected = "npu" if "hccl" in backend else "cuda"
    if resolved.type != expected:
        raise ValueError(f"{backend} control collectives require a {expected} device, got {resolved}.")
    return resolved


def _encode_control(value: Any, device: Any) -> tuple[torch.Tensor, int]:
    encoded = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    resolved = torch.device(device)
    payload = torch.zeros((max(1, len(encoded)),), dtype=torch.uint8, device=resolved)
    if encoded:
        payload[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=resolved)
    return payload, len(encoded)


def _decode_control(payload: torch.Tensor, size: int) -> Any:
    encoded = payload[:size].cpu().numpy().tobytes()
    return pickle.loads(encoded)


def all_gather_control_object(
        value: Any,
        *,
        group: Any = None,
        device: Any = None,
        backend: str | None = None,
) -> tuple[Any, ...]:
    """Gather a small control object with Gloo or an accelerator backend.

    Args:
        value: Rank-local serializable control value.
        group: Process group whose ranks participate in the gather.
        device: Rank-local accelerator used by HCCL/NCCL.
        backend: Explicit backend for startup calls that use WORLD.

    Returns:
        Gathered values in process-group rank order.
    """
    effective_backend = _control_backend(group, backend)
    if not _is_accelerator_backend(effective_backend):
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, value, group=group)
        return tuple(gathered)
    resolved = _require_control_device(device, effective_backend)
    encoded, encoded_size = _encode_control(value, resolved)
    local_size = torch.tensor([encoded_size], dtype=torch.int64, device=resolved)
    sizes = [torch.empty_like(local_size) for _ in range(dist.get_world_size(group))]
    dist.all_gather(sizes, local_size, group=group)
    max_size = max(int(item.item()) for item in sizes)
    if encoded.numel() < max_size:
        padded = torch.zeros((max_size,), dtype=torch.uint8, device=resolved)
        padded[:encoded.numel()] = encoded
        encoded = padded
    gathered = [torch.empty_like(encoded) for _ in sizes]
    dist.all_gather(gathered, encoded, group=group)
    return tuple(_decode_control(payload, int(size.item())) for payload, size in zip(gathered, sizes))


def broadcast_control_object(
        value: Any,
        *,
        src: int,
        group: Any = None,
        device: Any = None,
) -> Any:
    """Broadcast a small control object with Gloo or an accelerator backend.

    Args:
        value: Source value or a placeholder on receivers.
        src: Global source rank.
        group: Process group whose ranks participate in the broadcast.
        device: Rank-local accelerator used by HCCL/NCCL.

    Returns:
        The source value on every participating rank.
    """
    effective_backend = _control_backend(group)
    if not _is_accelerator_backend(effective_backend):
        payload = [value]
        dist.broadcast_object_list(payload, src=src, group=group)
        return payload[0]
    resolved = _require_control_device(device, effective_backend)
    rank = dist.get_rank()
    encoded, encoded_size = _encode_control(value, resolved) if rank == src else (
        torch.zeros((1,), dtype=torch.uint8, device=resolved), 0
    )
    size = torch.tensor([encoded_size], dtype=torch.int64, device=resolved)
    dist.broadcast(size, src=src, group=group)
    target_size = int(size.item())
    if encoded.numel() < max(1, target_size):
        encoded = torch.zeros((max(1, target_size),), dtype=torch.uint8, device=resolved)
    dist.broadcast(encoded, src=src, group=group)
    return _decode_control(encoded, target_size)


def synchronize_build_preflight(
        *,
        build_fingerprint: str | None,
        is_reader: bool,
        reader_size: int | None,
        is_direct_reader: bool,
        direct_dataset_size: int | None,
        metadata_mode: bool,
        dataset_already_sharded: bool,
        local_error: str | None,
        external_step_mode: bool = False,
        communication_backend: str = "hccl",
        communication_device: Any = None,
) -> bool:
    """Validate rank-local build inputs on WORLD before creating subgroups.

    Args:
        build_fingerprint: Stable topology and configuration identity, or
            ``None`` when local validation failed.
        is_reader: Whether this WORLD rank is configured as a Dataset Reader.
        reader_size: Dataset length in online mode or metadata length in
            metadata mode on Dataset Reader ranks.
        is_direct_reader: Whether this rank reads samples selected from metadata.
        direct_dataset_size: Mapping-Dataset length on plan-aware reader ranks.
        metadata_mode: Whether metadata is available before sample reads.
        dataset_already_sharded: Whether each Dataset Reader owns an independent
            local sample and metadata stream.
        local_error: Formatted local validation error, if any.
        external_step_mode: Whether this rank owns an external step reader.

    Returns:
        Whether all Dataset Readers use external step selection.

    Raises:
        ValueError: If any rank failed validation, build inputs differ, or
            Dataset Readers and direct readers do not expose one consistent
            logical index space.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    status = (
        rank,
        build_fingerprint,
        is_reader,
        reader_size,
        is_direct_reader,
        direct_dataset_size,
        metadata_mode,
        dataset_already_sharded,
        local_error,
        external_step_mode,
    )
    if not distributed:
        if local_error is not None:
            raise ValueError(f"Distributed DataLoader build preflight failed on rank {rank}: {local_error}")
        return external_step_mode

    startup_group = None
    if communication_backend == "gloo":
        startup_group = dist.new_group(ranks=list(range(dist.get_world_size())), backend="gloo")
    gathered = list(all_gather_control_object(
        status,
        group=startup_group,
        device=communication_device,
        backend=communication_backend,
    ))
    _validate_build_errors_and_fingerprint(gathered)
    _validate_build_modes(gathered)
    dataset_reader_sizes = [(item[0], item[3]) for item in gathered if item[2]]
    _validate_dataset_reader_sizes(
        dataset_reader_sizes,
        metadata_mode=metadata_mode,
        dataset_already_sharded=dataset_already_sharded,
    )
    direct_reader_sizes = [(item[0], item[5]) for item in gathered if item[4]]
    _validate_direct_reader_sizes(direct_reader_sizes, dataset_already_sharded=dataset_already_sharded)
    _validate_metadata_size_alignment(
        dataset_reader_sizes,
        direct_reader_sizes,
        metadata_mode=metadata_mode,
        dataset_already_sharded=dataset_already_sharded,
    )
    reader_modes = {item[9] for item in gathered if item[2]}
    if len(reader_modes) > 1:
        raise ValueError("Distributed DataLoader external step mode differs across Dataset Readers.")
    return True in reader_modes


def _validate_build_errors_and_fingerprint(normalized: Sequence[tuple[Any, ...]]) -> None:
    errors = [(item[0], item[8]) for item in normalized if item[8] is not None]
    if errors:
        error_rank, error = min(errors)
        raise ValueError(f"Distributed DataLoader build preflight failed on rank {error_rank}: {error}")

    fingerprints = {item[1] for item in normalized}
    if len(fingerprints) != 1 or None in fingerprints:
        details = ", ".join(f"rank {item[0]}={item[1]!r}" for item in normalized)
        raise ValueError(f"Distributed DataLoader build configuration mismatch across WORLD ranks: {details}.")


def _validate_build_modes(normalized: Sequence[tuple[Any, ...]]) -> None:
    modes = {item[6] for item in normalized}
    if len(modes) != 1:
        raise ValueError("Distributed DataLoader metadata mode differs across WORLD ranks.")
    sharding_modes = {item[7] for item in normalized}
    if len(sharding_modes) != 1:
        raise ValueError("Distributed DataLoader Dataset sharding mode differs across WORLD ranks.")


def _validate_dataset_reader_sizes(
        dataset_reader_sizes: Sequence[tuple[int, int | None]],
        *,
        metadata_mode: bool,
        dataset_already_sharded: bool,
) -> None:
    invalid_dataset_reader_sizes = [
        (reader_rank, size)
        for reader_rank, size in dataset_reader_sizes
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0)
    ]
    if invalid_dataset_reader_sizes:
        reader_data_name = "Metadata" if metadata_mode else "Dataset"
        raise ValueError(
            f"{reader_data_name} length is invalid on Dataset Reader ranks {invalid_dataset_reader_sizes}."
        )
    if metadata_mode and any(size is None for _, size in dataset_reader_sizes):
        raise ValueError("Metadata must have a finite length on every Dataset Reader rank.")
    if not dataset_already_sharded and len({size for _, size in dataset_reader_sizes}) > 1:
        reader_data_name = "Metadata" if metadata_mode else "Dataset"
        raise ValueError(
            f"{reader_data_name} length mismatch across Dataset Reader ranks: {dataset_reader_sizes}."
        )


def _validate_direct_reader_sizes(
        direct_reader_sizes: Sequence[tuple[int, int | None]],
        *,
        dataset_already_sharded: bool,
) -> None:
    invalid_direct_reader_sizes = [
        (reader_rank, size)
        for reader_rank, size in direct_reader_sizes
        if not isinstance(size, int) or isinstance(size, bool) or size < 0
    ]
    if invalid_direct_reader_sizes:
        raise ValueError(f"Direct-reader Dataset length is invalid on ranks {invalid_direct_reader_sizes}.")
    if not dataset_already_sharded and len({size for _, size in direct_reader_sizes}) > 1:
        raise ValueError(f"Direct-reader Dataset length mismatch across constructor ranks: {direct_reader_sizes}.")


def _validate_metadata_size_alignment(
        dataset_reader_sizes: Sequence[tuple[int, int | None]],
        direct_reader_sizes: Sequence[tuple[int, int | None]],
        *,
        metadata_mode: bool,
        dataset_already_sharded: bool,
) -> None:
    if not metadata_mode:
        return
    if dataset_already_sharded:
        metadata_sizes = dict(dataset_reader_sizes)
        sample_sizes = dict(direct_reader_sizes)
        aligned = metadata_sizes.keys() == sample_sizes.keys() and all(
            metadata_sizes[rank] == sample_sizes[rank] for rank in metadata_sizes
        )
        if not aligned:
            raise ValueError(
                "Pre-sharded metadata and Dataset lengths must match on each Dataset Reader rank, "
                f"but got metadata_readers={dataset_reader_sizes}, sample_readers={direct_reader_sizes}."
            )
        return
    dataset_reader_lengths = {size for _, size in dataset_reader_sizes}
    direct_reader_lengths = {size for _, size in direct_reader_sizes}
    aligned = (
        len(dataset_reader_lengths) == 1
        and len(direct_reader_lengths) == 1
        and dataset_reader_lengths == direct_reader_lengths
    )
    if not aligned:
        raise ValueError(
            f"Metadata and direct-reader Dataset lengths must match, but got "
            f"dataset_readers={dataset_reader_sizes}, direct_readers={direct_reader_sizes}."
        )


def create_data_groups(
        topology: DataTopology,
        dataset_reader_ranks: tuple[int, ...],
        planner_rank: int,
        *,
        cpu_backend: str,
        payload_backend: str | None,
        communication_device: Any,
        enable_payload_exchange: bool,
) -> DataGroups:
    """Create control, optional payload, and model groups deterministically.

    Args:
        topology: Constructor and model-consumer topology.
        dataset_reader_ranks: Global ranks that materialize raw samples.
        planner_rank: Global rank that creates loading plans.
        cpu_backend: Backend used for control and model-parallel batch broadcast.
        payload_backend: Optional backend for online payload A2A. When omitted
            with an accelerator communication device, the WORLD backend is used.
        communication_device: Optional rank-local device for payload tensors.
        enable_payload_exchange: Whether the online path needs a payload group.

    Returns:
        Groups relevant to the current rank.
    """
    data_plane_ranks = tuple(sorted(set(dataset_reader_ranks) | set(topology.constructor_ranks)))
    if planner_rank not in data_plane_ranks:
        raise ValueError(f"planner_rank {planner_rank} must belong to data-plane ranks {data_plane_ranks}.")
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed:
        return _single_rank_data_groups(topology, data_plane_ranks, planner_rank)
    _validate_distributed_topology(topology)
    _validate_group_backends(cpu_backend, payload_backend, enable_payload_exchange)
    effective_payload_backend, reuse_control_group = _resolve_payload_backend(
        data_plane_ranks=data_plane_ranks,
        cpu_backend=cpu_backend,
        payload_backend=payload_backend,
        communication_device=communication_device,
        enable_payload_exchange=enable_payload_exchange,
    )
    control_group, payload_group = _create_data_plane_process_groups(
        topology=topology,
        data_plane_ranks=data_plane_ranks,
        cpu_backend=cpu_backend,
        payload_backend=effective_payload_backend,
        reuse_control_group=reuse_control_group,
        enable_payload_exchange=enable_payload_exchange,
    )
    model_parallel_group, model_parallel_tensor_group = _create_model_parallel_process_groups(
        topology, cpu_backend, _resolve_model_tensor_backend(communication_device, payload_backend),
    )
    return DataGroups(
        data_plane_ranks, control_group, payload_group, model_parallel_group, planner_rank, True,
        model_parallel_tensor_group,
    )


def _single_rank_data_groups(
        topology: DataTopology,
        data_plane_ranks: tuple[int, ...],
        planner_rank: int,
) -> DataGroups:
    if len(topology.rank_list) != 1:
        raise ValueError("torch.distributed must be initialized for a mesh containing more than one rank.")
    return DataGroups(data_plane_ranks, None, None, None, planner_rank, False, None)


def _validate_distributed_topology(topology: DataTopology) -> None:
    world_ranks = tuple(range(dist.get_world_size()))
    if set(topology.rank_list) != set(world_ranks):
        raise ValueError(
            "The distributed data root mesh must contain every torch.distributed world rank so all ranks create "
            "service groups in the same order."
        )
    if dist.get_rank() != topology.global_rank:
        raise ValueError("Topology global_rank does not match torch.distributed rank.")


def _validate_group_backends(
        cpu_backend: str,
        payload_backend: str | None,
        enable_payload_exchange: bool,
) -> None:
    if not isinstance(cpu_backend, str) or not cpu_backend:
        raise ValueError("cpu_backend must be a non-empty backend name.")
    if cpu_backend.lower() not in ("gloo", "hccl"):
        raise ValueError("cpu_backend must be 'gloo' or 'hccl'.")
    if payload_backend is not None and (not isinstance(payload_backend, str) or not payload_backend):
        raise ValueError("payload_backend must be a non-empty backend name or None.")
    if not isinstance(enable_payload_exchange, bool):
        raise ValueError("enable_payload_exchange must be boolean.")


def _resolve_payload_backend(
        *,
        data_plane_ranks: tuple[int, ...],
        cpu_backend: str,
        payload_backend: str | None,
        communication_device: Any,
        enable_payload_exchange: bool,
) -> tuple[str | None, bool]:
    if not enable_payload_exchange or len(data_plane_ranks) == 1:
        return None, False
    device = torch.device(communication_device) if communication_device is not None else None
    effective_backend = payload_backend or _default_payload_backend(device, cpu_backend)
    normalized_backend = effective_backend.lower()
    _validate_payload_device(normalized_backend, effective_backend, device)
    reuse_control_group = normalized_backend == cpu_backend.lower() and (device is None or device.type == "cpu")
    return effective_backend, reuse_control_group


def _default_payload_backend(device: torch.device | None, cpu_backend: str) -> str:
    if device is not None and device.type != "cpu":
        return str(dist.get_backend())
    return cpu_backend


def _validate_payload_device(
        normalized_backend: str,
        effective_backend: str,
        device: torch.device | None,
) -> None:
    accelerator_backend = "hccl" in normalized_backend or "nccl" in normalized_backend
    if accelerator_backend and (device is None or device.type == "cpu"):
        raise ValueError(
            f"Payload backend {effective_backend!r} requires a rank-local accelerator communication_device."
        )
    expected_device_type = None
    if "nccl" in normalized_backend:
        expected_device_type = "cuda"
    elif "hccl" in normalized_backend:
        expected_device_type = "npu"
    if expected_device_type is not None and device is not None and device.type != expected_device_type:
        raise ValueError(
            f"Payload backend {effective_backend!r} requires a {expected_device_type} device, but got {device}."
        )
    if not accelerator_backend and device is not None and device.type != "cpu":
        raise ValueError(f"Payload backend {effective_backend!r} cannot exchange tensors on device {device}.")


def _create_data_plane_process_groups(
        *,
        topology: DataTopology,
        data_plane_ranks: tuple[int, ...],
        cpu_backend: str,
        payload_backend: str | None,
        reuse_control_group: bool,
        enable_payload_exchange: bool,
) -> tuple[Any, Any]:
    if len(data_plane_ranks) == 1:
        return None, None
    created_control_group = dist.new_group(ranks=list(data_plane_ranks), backend=cpu_backend)
    is_member = topology.global_rank in data_plane_ranks
    control_group = created_control_group if is_member else None
    if not enable_payload_exchange:
        return control_group, None
    if reuse_control_group:
        return control_group, control_group
    created_payload_group = dist.new_group(ranks=list(data_plane_ranks), backend=payload_backend)
    return control_group, created_payload_group if is_member else None


def _resolve_model_tensor_backend(communication_device: Any, payload_backend: str | None) -> str | None:
    """Select an accelerator backend for tensor model-group broadcasts when available."""
    if communication_device is None:
        return None
    device = torch.device(communication_device)
    if device.type == "cpu":
        return None
    candidate = payload_backend or str(dist.get_backend())
    normalized = candidate.lower()
    if "hccl" not in normalized and "nccl" not in normalized:
        return None
    return candidate


def _create_model_parallel_process_groups(
        topology: DataTopology,
        cpu_backend: str,
        tensor_backend: str | None,
) -> tuple[Any, Any]:
    model_parallel_group = None
    model_parallel_tensor_group = None
    for rank_group in topology.model_parallel_rank_groups:
        if len(rank_group) == 1:
            continue
        created_group = dist.new_group(ranks=list(rank_group), backend=cpu_backend)
        if topology.global_rank in rank_group:
            model_parallel_group = created_group
        if tensor_backend is not None:
            created_tensor_group = dist.new_group(ranks=list(rank_group), backend=tensor_backend)
            if topology.global_rank in rank_group:
                model_parallel_tensor_group = created_tensor_group
    return model_parallel_group, model_parallel_tensor_group


def _encode_payload_segment(items: Sequence[tuple[SampleKey, Any]]) -> bytes:
    """Serialize an internal route; the collective backend transports its bytes."""
    if not items:
        return b""
    return pickle.dumps(tuple(items), protocol=pickle.HIGHEST_PROTOCOL)


def _decode_payload_segment(frame: bytes) -> tuple[tuple[SampleKey, Any], ...]:
    """Deserialize a route produced by peers running the same codec."""
    if not frame:
        return ()
    return pickle.loads(frame)


def _encode_model_batch(value: Any) -> tuple[Any, list[torch.Tensor]]:
    """Replace tensor leaves with descriptors before object broadcast."""
    tensors: list[torch.Tensor] = []

    def _encode(item: Any) -> Any:
        if torch.is_tensor(item):
            index = len(tensors)
            tensors.append(item)
            return ("__hp_tensor__", index, tuple(item.shape), item.dtype, item.device.type)
        if isinstance(item, Mapping):
            return ("__hp_mapping__", tuple((key, _encode(child)) for key, child in item.items()))
        if isinstance(item, tuple):
            return ("__hp_tuple__", tuple(_encode(child) for child in item))
        if isinstance(item, list):
            return ("__hp_list__", tuple(_encode(child) for child in item))
        return ("__hp_value__", item)

    return _encode(value), tensors


def _decode_model_batch(schema: Any, tensors: Sequence[torch.Tensor]) -> Any:
    """Reconstruct a batch from a broadcast structure and tensor leaves."""
    kind = schema[0]
    if kind == "__hp_tensor__":
        return tensors[schema[1]]
    if kind == "__hp_mapping__":
        return {key: _decode_model_batch(child, tensors) for key, child in schema[1]}
    if kind == "__hp_tuple__":
        return tuple(_decode_model_batch(child, tensors) for child in schema[1])
    if kind == "__hp_list__":
        return [_decode_model_batch(child, tensors) for child in schema[1]]
    if kind == "__hp_value__":
        return schema[1]
    raise ValueError(f"Model batch broadcast received unknown descriptor kind {kind!r}.")


def _tensor_specs(schema: Any) -> list[tuple[tuple[int, ...], torch.dtype, str]]:
    """Collect tensor descriptors in the same order used by the encoder."""
    kind = schema[0]
    if kind == "__hp_tensor__":
        return [(schema[2], schema[3], schema[4])]
    if kind == "__hp_mapping__":
        children = (child for _, child in schema[1])
    elif kind in ("__hp_tuple__", "__hp_list__"):
        children = iter(schema[1])
    elif kind == "__hp_value__":
        return []
    else:
        raise ValueError(f"Model batch broadcast received unknown descriptor kind {kind!r}.")
    specs = []
    for child in children:
        specs.extend(_tensor_specs(child))
    return specs


def _tensor_device(device_type: str, group: Any) -> torch.device:
    """Resolve the receiver's local device for a tensor model-group broadcast."""
    backend = str(dist.get_backend(group)).lower()
    if "hccl" in backend:
        expected_type = "npu"
    elif "nccl" in backend:
        expected_type = "cuda"
    else:
        expected_type = "cpu"
    if device_type != expected_type:
        raise ValueError(
            f"Model batch tensor device {device_type!r} is incompatible with {backend!r} model broadcast."
        )
    if expected_type == "cpu":
        return torch.device("cpu")
    module = torch.npu if expected_type == "npu" else torch.cuda
    return torch.device(expected_type, module.current_device())


def _exchange_payload_sizes(
        input_splits: Sequence[int],
        control_group: Any,
        device: Any = None,
) -> list[int]:
    backend = _control_backend(control_group)
    size_device = _require_control_device(device, backend) if _is_accelerator_backend(backend) else torch.device("cpu")
    size_input = torch.tensor(tuple(input_splits), dtype=torch.int64, device=size_device)
    size_output = torch.empty_like(size_input)
    size_work = dist.all_to_all_single(size_output, size_input, group=control_group, async_op=True)
    if size_work is not None:
        size_work.wait()
    return [int(size) for size in size_output.tolist()]


def _decode_received_payloads(
        received_bytes: bytes,
        output_splits: Sequence[int],
) -> dict[SampleKey, Any]:
    """Merge routes without silently overwriting duplicate sample occurrences."""
    payloads: dict[SampleKey, Any] = {}
    cursor = 0
    for segment_size in output_splits:
        segment_items = _decode_payload_segment(received_bytes[cursor:cursor + segment_size])
        for key, payload in segment_items:
            if key in payloads:
                raise ValueError(f"Data Constructor received duplicate payload for {key}.")
            payloads[key] = payload
        cursor += segment_size
    return payloads


class DataPlaneTransport:
    """Metadata/control collectives and variable-byte sample all-to-all.

    The coordinator calls these methods only on data-plane members. Group
    topology is fixed at build time; it is not revalidated for each collective.
    """

    def __init__(self, groups: DataGroups, global_rank: int, communication_device: Any = None) -> None:
        """Store service-group membership."""
        self._ranks = groups.data_plane_ranks
        self._control_group = groups.control_group
        self._payload_group = groups.payload_group
        self._planner_rank = groups.planner_rank
        self._global_rank = global_rank
        self._is_member = global_rank in self._ranks
        self._local_index = self._ranks.index(global_rank) if self._is_member else None
        if self._is_member and len(self._ranks) > 1 and (not groups.distributed or self._control_group is None):
            raise ValueError("Multi-rank data-plane communication requires an initialized process group.")
        self._communication_device = (
            torch.device(communication_device) if communication_device is not None else None
        )

    @property
    def is_member(self) -> bool:
        """Return whether this rank participates in Dataset Reader/Planner communication."""
        return self._is_member

    @property
    def planner_rank(self) -> int:
        """Return the global Planner rank."""
        return self._planner_rank

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return data-plane ranks in collective order."""
        return self._ranks

    @property
    def communication_device(self) -> torch.device | None:
        """Return the rank-local accelerator used by payload collectives."""
        return self._communication_device

    @property
    def communication_backend(self) -> str:
        """Return the backend used by this data-plane control group."""
        return _control_backend(self._control_group)

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Gather small control objects on every data-plane rank.

        Args:
            value: Rank-local control value.

        Returns:
            Values in data-plane rank order.
        """
        if len(self._ranks) == 1:
            return (value,)
        return all_gather_control_object(
            value,
            group=self._control_group,
            device=self._communication_device,
        )

    def all_ranks_true(self, value: bool) -> bool:
        """Return whether every data-plane rank supplied ``True``.

        Args:
            value: Rank-local boolean value.

        Returns:
            Whether every participating rank supplied ``True``.
        """
        if len(self._ranks) == 1:
            return value
        backend = _control_backend(self._control_group)
        flag_device = (
            _require_control_device(self._communication_device, backend)
            if _is_accelerator_backend(backend)
            else torch.device("cpu")
        )
        flag = torch.tensor([int(value)], dtype=torch.int32, device=flag_device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._control_group)
        return bool(flag.item())

    def gather_object_to_planner(self, value: Any) -> tuple[Any, ...] | None:
        """Gather Dataset Reader metadata only on the Planner rank.

        Args:
            value: Rank-local metadata control value.

        Returns:
            Gathered values on the Planner rank, otherwise ``None``.
        """
        if len(self._ranks) == 1:
            return (value,)
        gathered = all_gather_control_object(
            value,
            group=self._control_group,
            device=self._communication_device,
        )
        return gathered if self._global_rank == self._planner_rank else None

    def broadcast_from_planner(self, value: Any) -> Any:
        """Broadcast one control object from the configured Planner.

        Args:
            value: Planner value or a placeholder on other ranks.

        Returns:
            The Planner value on every data-plane rank.
        """
        if len(self._ranks) == 1:
            return value
        return broadcast_control_object(
            value if self._global_rank == self._planner_rank else None,
            src=self._planner_rank,
            group=self._control_group,
            device=self._communication_device,
        )

    def prepare_exchange(
            self,
            outgoing: Mapping[int, Sequence[tuple[SampleKey, Any]]],
    ) -> PreparedPayloadExchange:
        """Serialize and allocate the send buffer before collective entry.

        Args:
            outgoing: Per-target sample keys and payloads.

        Returns:
            Serialized payload splits and their send tensor.
        """
        unexpected = outgoing.keys() - self._ranks
        if unexpected:
            raise ValueError(f"Payload routes target ranks outside the data plane: {sorted(unexpected)}.")
        segments = tuple(_encode_payload_segment(outgoing.get(rank, ())) for rank in self._ranks)
        input_splits = tuple(len(segment) for segment in segments)
        send_storage = bytearray(sum(input_splits))
        cursor = 0
        for segment in segments:
            send_storage[cursor:cursor + len(segment)] = segment
            cursor += len(segment)
        if send_storage:
            send_tensor = torch.frombuffer(send_storage, dtype=torch.uint8)
        else:
            send_tensor = torch.empty((0,), dtype=torch.uint8)
        if self._communication_device is not None:
            send_tensor = send_tensor.to(self._communication_device)
        return PreparedPayloadExchange(
            input_splits=input_splits,
            send_storage=send_storage,
            send_tensor=send_tensor,
            local_segment=segments[self._local_index],
        )

    def exchange_prepared(
            self,
            prepared: PreparedPayloadExchange,
    ) -> dict[SampleKey, Any]:
        """Exchange serialized sample payloads with variable-split payload A2A.

        Args:
            prepared: Preallocated payload exchange state.

        Returns:
            Received payloads keyed by their source identities.
        """
        if len(self._ranks) == 1:
            return _decode_received_payloads(prepared.local_segment, prepared.input_splits)
        if self._payload_group is None:
            raise ValueError("Sample payload exchange requires a payload process group.")

        input_splits = list(prepared.input_splits)
        output_splits = _exchange_payload_sizes(
            input_splits,
            self._control_group,
            device=self._communication_device,
        )
        received_tensor = torch.empty(
            (sum(output_splits),),
            dtype=torch.uint8,
            device=prepared.send_tensor.device,
        )
        data_work = dist.all_to_all_single(
            received_tensor,
            prepared.send_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self._payload_group,
            async_op=True,
        )
        if data_work is not None:
            data_work.wait()
        received_bytes = received_tensor.cpu().numpy().tobytes()
        return _decode_received_payloads(received_bytes, output_splits)


class ModelParallelTransport:
    """Broadcast a constructed batch with direct tensor leaves when possible."""

    def __init__(self, topology: DataTopology, groups: DataGroups, communication_device: Any = None) -> None:
        """Store the current model-consumer group."""
        self._ranks = topology.model_parallel_ranks
        self._object_group = groups.model_parallel_group
        self._accelerator_tensor_group = groups.model_parallel_tensor_group
        self._constructor_rank = topology.constructor_rank
        self._global_rank = topology.global_rank
        self._communication_device = (
            torch.device(communication_device) if communication_device is not None else None
        )
        if len(self._ranks) > 1 and (not groups.distributed or self._object_group is None):
            raise ValueError("Multi-rank model broadcast requires an initialized process group.")

    def broadcast(self, batch: Any) -> Any:
        """Return the constructor's batch data on every model-parallel peer.

        ``None`` is reserved as the end-of-stream sentinel. Standard dict,
        list, and tuple containers retain their structure while tensor leaves
        use a direct tensor broadcast instead of pickle serialization.

        Args:
            batch: Constructor batch or ``None`` on consumer-only peers.

        Returns:
            The constructor's batch data, or ``None`` at end of stream.
        """
        is_constructor = self._global_rank == self._constructor_rank
        if len(self._ranks) == 1:
            return batch
        schema, source_tensors = _encode_model_batch(batch) if is_constructor else (None, [])
        backend = _control_backend(self._object_group)
        if _is_accelerator_backend(backend):
            received_schema = broadcast_control_object(
                schema if is_constructor else None,
                src=self._constructor_rank,
                group=self._object_group,
                device=self._communication_device,
            )
        else:
            payload = [schema]
            dist.broadcast_object_list(payload, src=self._constructor_rank, group=self._object_group)
            received_schema = payload[0]
        tensor_specs = _tensor_specs(received_schema)
        received_tensors = []
        for index, spec in enumerate(tensor_specs):
            tensor_group = self._group_for_tensor(spec[2])
            device = _tensor_device(spec[2], tensor_group)
            if is_constructor:
                tensor = source_tensors[index]
            else:
                tensor = torch.empty(spec[0], dtype=spec[1], device=device)
            dist.broadcast(tensor, src=self._constructor_rank, group=tensor_group)
            received_tensors.append(tensor)
        return batch if is_constructor else _decode_model_batch(received_schema, received_tensors)

    def _group_for_tensor(self, device_type: str) -> Any:
        """Choose Gloo for host tensors and the accelerator group for device tensors."""
        if device_type == "cpu":
            return self._object_group
        if self._accelerator_tensor_group is None:
            return self._object_group
        return self._accelerator_tensor_group


__all__ = [
    "DataGroups",
    "DataPlaneTransport",
    "ModelParallelTransport",
    "PreparedPayloadExchange",
    "create_data_groups",
    "synchronize_build_preflight",
]
