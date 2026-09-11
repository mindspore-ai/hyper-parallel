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
"""CPU control and configurable payload data planes for dynamic packing."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import ConstructedBatch, SampleKey
from hyper_parallel.distributed_data.topology import DataTopology

_FRAME_MAGIC = b"HPDDP1"
_DIGEST_SIZE = 32


@dataclass(frozen=True)
class DataGroups:
    """Control, payload, and model-consumer groups for distributed data."""

    data_plane_ranks: tuple[int, ...]
    control_group: Any
    payload_group: Any
    model_parallel_group: Any
    planner_rank: int
    distributed: bool


@dataclass(frozen=True)
class PreparedPayloadExchange:
    """Preflighted Host frame and collective tensor for one payload A2A."""

    input_splits: tuple[int, ...]
    send_storage: bytearray
    send_tensor: torch.Tensor
    local_segment: bytes


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
) -> None:
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
    )
    if not distributed:
        if local_error is not None:
            raise ValueError(f"Distributed DataLoader build preflight failed on rank {rank}: {local_error}")
        return

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, status)
    normalized = _normalize_build_statuses(gathered)
    _validate_build_errors_and_fingerprint(normalized)
    _validate_build_modes(normalized)
    dataset_reader_sizes = [(item[0], item[3]) for item in normalized if item[2]]
    _validate_dataset_reader_sizes(
        dataset_reader_sizes,
        metadata_mode=metadata_mode,
        dataset_already_sharded=dataset_already_sharded,
    )
    direct_reader_sizes = [(item[0], item[5]) for item in normalized if item[4]]
    _validate_direct_reader_sizes(direct_reader_sizes, dataset_already_sharded=dataset_already_sharded)
    _validate_metadata_size_alignment(
        dataset_reader_sizes,
        direct_reader_sizes,
        metadata_mode=metadata_mode,
        dataset_already_sharded=dataset_already_sharded,
    )


def _normalize_build_statuses(gathered: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    normalized = []
    for expected_rank, item in enumerate(gathered):
        if not isinstance(item, tuple) or len(item) != 9 or item[0] != expected_rank:
            raise ValueError("Distributed DataLoader build preflight received an invalid WORLD status.")
        normalized.append(item)
    return tuple(normalized)


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
        cpu_backend: Backend used for control and object delivery.
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
    model_parallel_group = _create_model_parallel_process_groups(topology, cpu_backend)
    return DataGroups(data_plane_ranks, control_group, payload_group, model_parallel_group, planner_rank, True)


def _single_rank_data_groups(
        topology: DataTopology,
        data_plane_ranks: tuple[int, ...],
        planner_rank: int,
) -> DataGroups:
    if len(topology.rank_list) != 1:
        raise ValueError("torch.distributed must be initialized for a mesh containing more than one rank.")
    return DataGroups(data_plane_ranks, None, None, None, planner_rank, False)


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
    if "hccl" in cpu_backend.lower() or "nccl" in cpu_backend.lower():
        raise ValueError("cpu_backend must support CPU tensors and object collectives; use Gloo, not HCCL/NCCL.")
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


def _create_model_parallel_process_groups(topology: DataTopology, cpu_backend: str) -> Any:
    model_parallel_group = None
    for rank_group in topology.model_parallel_rank_groups:
        if len(rank_group) == 1:
            continue
        created_group = dist.new_group(ranks=list(rank_group), backend=cpu_backend)
        if topology.global_rank in rank_group:
            model_parallel_group = created_group
    return model_parallel_group


def _encode_payload_segment(items: Sequence[tuple[SampleKey, Any]]) -> bytes:
    """Serialize one target route with an integrity digest."""
    if not items:
        return b""
    keys = [key for key, _ in items]
    if any(not isinstance(key, SampleKey) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("A payload segment must contain unique SampleKey values.")
    payload = pickle.dumps(tuple(items), protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).digest()
    return _FRAME_MAGIC + digest + payload


def _decode_payload_segment(frame: bytes) -> tuple[tuple[SampleKey, Any], ...]:
    """Validate and deserialize one target route."""
    if not frame:
        return ()
    header_size = len(_FRAME_MAGIC) + _DIGEST_SIZE
    if len(frame) < header_size or frame[:len(_FRAME_MAGIC)] != _FRAME_MAGIC:
        raise ValueError("Distributed sample payload has an invalid frame header.")
    expected_digest = frame[len(_FRAME_MAGIC):header_size]
    payload = frame[header_size:]
    if hashlib.sha256(payload).digest() != expected_digest:
        raise ValueError("Distributed sample payload checksum mismatch.")
    try:
        items = pickle.loads(payload)
    except Exception as exc:
        raise ValueError(f"Distributed sample payload cannot be decoded: {exc}") from exc
    if not isinstance(items, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], SampleKey)
            for item in items
    ):
        raise ValueError("Distributed sample payload contains an invalid route segment.")
    keys = [key for key, _ in items]
    if len(keys) != len(set(keys)):
        raise ValueError("Distributed sample payload contains duplicate SampleKey values.")
    return items


def _is_non_negative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_iterator_fields(
        status: Any,
        ranks: Sequence[int],
        *,
        expected_length: int,
        scope: str,
) -> tuple[Any, ...] | str:
    if not isinstance(status, tuple) or len(status) != expected_length:
        return f"{scope} contributed an invalid iterator-state status."
    rank, rank_epoch, rank_step, rank_stopped = status[:4]
    if not _is_non_negative_integer(rank) or rank not in ranks:
        return f"{scope} contributed an invalid iterator-state rank."
    if not _is_non_negative_integer(rank_epoch):
        return f"{scope} {rank} contributed an invalid iterator epoch."
    if not _is_non_negative_integer(rank_step):
        return f"{scope} {rank} contributed an invalid iterator step."
    if not isinstance(rank_stopped, bool):
        return f"{scope} {rank} contributed an invalid stopped state."
    return status


def _normalize_iterator_statuses(
        statuses: Sequence[Any],
        ranks: Sequence[int],
        *,
        expected_length: int,
        scope: str,
) -> tuple[tuple[tuple[Any, ...], ...] | None, str | None]:
    normalized = []
    for status in statuses:
        result = _validate_iterator_fields(status, ranks, expected_length=expected_length, scope=scope)
        if isinstance(result, str):
            return None, result
        normalized.append(result)
    return tuple(normalized), None


def _iterator_coverage_error(
        normalized: Sequence[tuple[Any, ...]],
        ranks: Sequence[int],
        *,
        scope: str,
) -> str | None:
    contributed_ranks = [status[0] for status in normalized]
    if len(contributed_ranks) != len(set(contributed_ranks)) or set(contributed_ranks) != set(ranks):
        return f"{scope} iterator states have invalid rank coverage {contributed_ranks}."
    return None


def _iterator_state_error(normalized: Sequence[tuple[Any, ...]], *, scope: str) -> str | None:
    states = {(status[1], status[2], status[3]) for status in normalized}
    if len(states) == 1:
        return None
    details = ", ".join(
        f"rank {status[0]}=(epoch={status[1]}, step={status[2]}, stopped={status[3]})"
        for status in normalized
    )
    return f"{scope} ranks have inconsistent iterator state: {details}."


def _exchange_payload_sizes(input_splits: Sequence[int], control_group: Any) -> list[int]:
    size_input = torch.tensor(tuple(input_splits), dtype=torch.int64)
    size_output = torch.empty_like(size_input)
    size_work = dist.all_to_all_single(size_output, size_input, group=control_group, async_op=True)
    if size_work is not None:
        size_work.wait()
    return [int(size) for size in size_output.tolist()]


def _allocate_received_tensor(
        output_splits: Sequence[int],
        send_tensor: torch.Tensor,
        expected_split_count: int,
) -> tuple[torch.Tensor | None, str | None]:
    try:
        if len(output_splits) != expected_split_count or any(size < 0 for size in output_splits):
            raise ValueError(f"Sample all-to-all returned invalid payload sizes {list(output_splits)}.")
        received_tensor = torch.empty(
            (sum(output_splits),),
            dtype=torch.uint8,
            device=send_tensor.device,
        )
    except Exception as exc:
        return None, f"Payload receive allocation failed: {type(exc).__name__}: {exc}"
    return received_tensor, None


def _decode_received_payloads(received_bytes: bytes, output_splits: Sequence[int]) -> dict[SampleKey, Any]:
    payloads: dict[SampleKey, Any] = {}
    cursor = 0
    for segment_size in output_splits:
        segment_items = _decode_payload_segment(received_bytes[cursor:cursor + segment_size])
        for key, payload in segment_items:
            if key in payloads:
                raise ValueError(f"Data Constructor received duplicate payload for {key}.")
            payloads[key] = payload
        cursor += segment_size
    if cursor != len(received_bytes):
        raise ValueError("Sample all-to-all returned trailing payload bytes.")
    return payloads


class DataPlaneTransport:
    """Metadata/control collectives and variable-byte sample all-to-all."""

    def __init__(self, groups: DataGroups, global_rank: int, communication_device: Any = None) -> None:
        """Store service-group membership."""
        self._ranks = groups.data_plane_ranks
        self._control_group = groups.control_group
        self._payload_group = groups.payload_group
        self._planner_rank = groups.planner_rank
        self._distributed = groups.distributed
        self._global_rank = global_rank
        self._is_member = global_rank in self._ranks
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

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Gather small control objects on every data-plane rank.

        Args:
            value: Rank-local control value.

        Returns:
            Values in data-plane rank order.
        """
        self._require_member()
        if len(self._ranks) == 1:
            return (value,)
        gathered = [None] * len(self._ranks)
        dist.all_gather_object(gathered, value, group=self._control_group)
        return tuple(gathered)

    def gather_object_to_planner(self, value: Any) -> tuple[Any, ...] | None:
        """Gather Dataset Reader metadata only on the Planner rank.

        Args:
            value: Rank-local metadata control value.

        Returns:
            Gathered values on the Planner rank, otherwise ``None``.
        """
        self._require_member()
        if len(self._ranks) == 1:
            if self._global_rank != self._planner_rank:
                raise ValueError("A singleton data plane must contain the Planner rank.")
            return (value,)
        gathered = [None] * len(self._ranks) if self._global_rank == self._planner_rank else None
        dist.gather_object(
            value,
            object_gather_list=gathered,
            dst=self._planner_rank,
            group=self._control_group,
        )
        return tuple(gathered) if gathered is not None else None

    def broadcast_from_planner(self, value: Any) -> Any:
        """Broadcast one control object from the configured Planner.

        Args:
            value: Planner value or a placeholder on other ranks.

        Returns:
            The Planner value on every data-plane rank.
        """
        self._require_member()
        if len(self._ranks) == 1:
            if self._global_rank != self._planner_rank:
                raise ValueError("A singleton data plane must contain the Planner rank.")
            return value
        payload = [value if self._global_rank == self._planner_rank else None]
        dist.broadcast_object_list(payload, src=self._planner_rank, group=self._control_group)
        return payload[0]

    def synchronize_error(self, error: str | None) -> str | None:
        """Return the first rank-ordered error observed by the data plane.

        Args:
            error: Rank-local formatted error, if any.

        Returns:
            The first shared error in rank order, if any.
        """
        statuses = self.all_gather_object((self._global_rank, error))
        normalized = []
        for status in statuses:
            if not isinstance(status, tuple) or len(status) != 2:
                return "Data-plane rank contributed an invalid error status."
            rank, message = status
            if not isinstance(rank, int) or isinstance(rank, bool) or rank not in self._ranks:
                return "Data-plane rank contributed an invalid error-status rank."
            if message is not None and not isinstance(message, str):
                return f"Data-plane rank {rank} contributed a non-string error."
            if message is not None:
                normalized.append((rank, message))
        return min(normalized)[1] if normalized else None

    def synchronize_iterator_state(
            self,
            *,
            epoch: int,
            step: int,
            stopped: bool,
            model_group_error: str | None,
    ) -> str | None:
        """Validate iterator state across all model groups before reading.

        Args:
            epoch: Current epoch number.
            step: Current iterator step.
            stopped: Whether local iteration has stopped.
            model_group_error: Model-group synchronization error, if any.

        Returns:
            A shared validation error, if any.
        """
        statuses = self.all_gather_object((self._global_rank, epoch, step, stopped, model_group_error))
        normalized, validation_error = _normalize_iterator_statuses(
            statuses,
            self._ranks,
            expected_length=5,
            scope="Data-plane rank",
        )
        if validation_error is not None or normalized is None:
            return validation_error
        invalid_model_error = next(
            (status for status in normalized if status[4] is not None and not isinstance(status[4], str)),
            None,
        )
        if invalid_model_error is not None:
            return f"Data-plane rank {invalid_model_error[0]} contributed a non-string model-group error."
        coverage_error = _iterator_coverage_error(normalized, self._ranks, scope="Data-plane")
        if coverage_error is not None:
            return coverage_error
        errors = sorted((status[0], status[4]) for status in normalized if status[4] is not None)
        if errors:
            return errors[0][1]
        return _iterator_state_error(normalized, scope="Data-plane")

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
        self._require_member()
        unexpected = set(outgoing) - set(self._ranks)
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
        local_index = self._ranks.index(self._global_rank)
        return PreparedPayloadExchange(
            input_splits=input_splits,
            send_storage=send_storage,
            send_tensor=send_tensor,
            local_segment=segments[local_index],
        )

    def exchange_prepared(self, prepared: PreparedPayloadExchange) -> dict[SampleKey, Any]:
        """Exchange framed sample payloads with variable-split payload A2A.

        Args:
            prepared: Preallocated payload exchange state.

        Returns:
            Received payloads keyed by their source identities.
        """
        self._require_member()
        if not isinstance(prepared, PreparedPayloadExchange) or len(prepared.input_splits) != len(self._ranks):
            raise ValueError(f"Expected a prepared exchange for {len(self._ranks)} data-plane ranks.")
        if len(self._ranks) == 1:
            return dict(_decode_payload_segment(prepared.local_segment))
        if self._payload_group is None:
            raise ValueError("Sample payload exchange requires a payload process group.")

        input_splits = list(prepared.input_splits)
        output_splits = _exchange_payload_sizes(input_splits, self._control_group)
        received_tensor, allocation_error = _allocate_received_tensor(
            output_splits,
            prepared.send_tensor,
            len(self._ranks),
        )
        shared_error = self.synchronize_error(allocation_error)
        if shared_error is not None:
            raise ValueError(shared_error)
        if received_tensor is None:
            raise ValueError("Payload receive allocation did not produce a tensor.")
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

    def _require_member(self) -> None:
        if not self._is_member:
            raise ValueError(f"Global rank {self._global_rank} is not a data-plane member.")
        if len(self._ranks) > 1 and (not self._distributed or self._control_group is None):
            raise ValueError("Multi-rank data-plane communication requires an initialized process group.")


class ModelParallelTransport:
    """Broadcast a constructed CPU batch to peers in one model replica."""

    def __init__(self, topology: DataTopology, groups: DataGroups) -> None:
        """Store the current model-consumer group."""
        self._ranks = topology.model_parallel_ranks
        self._group = groups.model_parallel_group
        self._constructor_rank = topology.constructor_rank
        self._global_rank = topology.global_rank
        self._distributed = groups.distributed

    def synchronize_iterator_state(self, *, epoch: int, step: int, stopped: bool) -> str | None:
        """Return a shared error when peers restored different iterator state.

        Args:
            epoch: Current epoch number.
            step: Current iterator step.
            stopped: Whether local iteration has stopped.

        Returns:
            A shared validation error, if any.
        """
        if len(self._ranks) == 1:
            return None
        if not self._distributed or self._group is None:
            return "Multi-rank model delivery requires an initialized process group."
        gathered = [None] * len(self._ranks)
        dist.all_gather_object(gathered, (self._global_rank, epoch, step, stopped), group=self._group)
        normalized, validation_error = _normalize_iterator_statuses(
            gathered,
            self._ranks,
            expected_length=4,
            scope="Model-parallel rank",
        )
        if validation_error is not None or normalized is None:
            return validation_error
        coverage_error = _iterator_coverage_error(normalized, self._ranks, scope="Model-parallel")
        if coverage_error is not None:
            return coverage_error
        return _iterator_state_error(normalized, scope="Model-parallel")

    def broadcast(self, batch: ConstructedBatch | None) -> ConstructedBatch:
        """Return the constructor's batch envelope on every model-parallel peer.

        Args:
            batch: Constructor batch or ``None`` on consumer-only peers.

        Returns:
            The constructor's batch envelope.
        """
        is_constructor = self._global_rank == self._constructor_rank
        if len(self._ranks) == 1:
            if not is_constructor or not isinstance(batch, ConstructedBatch):
                raise ValueError("A singleton model group requires a local ConstructedBatch.")
            return batch
        if not self._distributed or self._group is None:
            raise ValueError("Multi-rank model delivery requires an initialized process group.")
        if is_constructor and not isinstance(batch, ConstructedBatch):
            raise ValueError("The Data Constructor must provide a ConstructedBatch.")
        if not is_constructor and batch is not None:
            raise ValueError("Only the Data Constructor may provide the model-group batch.")
        payload = [batch]
        dist.broadcast_object_list(payload, src=self._constructor_rank, group=self._group)
        received = payload[0]
        if not isinstance(received, ConstructedBatch):
            raise ValueError("Model-parallel broadcast did not contain a ConstructedBatch.")
        return received


__all__ = [
    "DataGroups",
    "DataPlaneTransport",
    "ModelParallelTransport",
    "PreparedPayloadExchange",
    "create_data_groups",
    "synchronize_build_preflight",
]
