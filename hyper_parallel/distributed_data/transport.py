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
"""CPU control and payload data planes for distributed dynamic packing."""
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
    """CPU service group and the current rank's model-consumer group."""

    data_plane_ranks: tuple[int, ...]
    data_plane_group: Any
    model_parallel_group: Any
    planner_rank: int
    distributed: bool


@dataclass(frozen=True)
class PreparedPayloadExchange:
    """Preflighted CPU send buffer for one payload all-to-all."""

    input_splits: tuple[int, ...]
    send_storage: bytearray
    send_tensor: torch.Tensor
    local_segment: bytes


def synchronize_build_preflight(
        *,
        build_fingerprint: str | None,
        is_source: bool,
        dataset_size: int | None,
        local_error: str | None,
) -> None:
    """Validate rank-local build inputs on WORLD before creating subgroups.

    Args:
        build_fingerprint: Stable topology and configuration identity, or
            ``None`` when local validation failed.
        is_source: Whether this WORLD rank is configured as a Source Loader.
        dataset_size: Logical mapping-Dataset length on Source ranks.
        local_error: Formatted local validation error, if any.

    Raises:
        ValueError: If any rank failed validation, build inputs differ, or
            Source ranks do not expose equal-length replicas of one Dataset.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    status = (rank, build_fingerprint, is_source, dataset_size, local_error)
    if not distributed:
        if local_error is not None:
            raise ValueError(f"Distributed DataLoader build preflight failed on rank {rank}: {local_error}")
        return

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, status)
    normalized = []
    for expected_rank, item in enumerate(gathered):
        if not isinstance(item, tuple) or len(item) != 5 or item[0] != expected_rank:
            raise ValueError("Distributed DataLoader build preflight received an invalid WORLD status.")
        normalized.append(item)

    errors = [(item[0], item[4]) for item in normalized if item[4] is not None]
    if errors:
        error_rank, error = min(errors)
        raise ValueError(f"Distributed DataLoader build preflight failed on rank {error_rank}: {error}")

    fingerprints = {item[1] for item in normalized}
    if len(fingerprints) != 1 or None in fingerprints:
        details = ", ".join(f"rank {item[0]}={item[1]!r}" for item in normalized)
        raise ValueError(f"Distributed DataLoader build configuration mismatch across WORLD ranks: {details}.")

    source_sizes = [(item[0], item[3]) for item in normalized if item[2]]
    invalid_sizes = [
        (source_rank, size)
        for source_rank, size in source_sizes
        if not isinstance(size, int) or isinstance(size, bool) or size < 0
    ]
    if invalid_sizes:
        raise ValueError(f"Source Dataset length is invalid on ranks {invalid_sizes}.")
    if len({size for _, size in source_sizes}) > 1:
        raise ValueError(
            f"Source Dataset length mismatch across replicated Source Loader ranks: {source_sizes}. "
            "Every Source Loader rank must receive the same logical mapping Dataset."
        )


def create_data_groups(
        topology: DataTopology,
        source_loader_ranks: tuple[int, ...],
        planner_rank: int,
        *,
        cpu_backend: str,
) -> DataGroups:
    """Create all CPU groups in one globally deterministic order.

    Args:
        topology: Constructor and model-consumer topology.
        source_loader_ranks: Global ranks that materialize raw samples.
        planner_rank: Global rank that creates loading plans.
        cpu_backend: PyTorch backend used for CPU communication.

    Returns:
        Groups relevant to the current rank.
    """
    data_plane_ranks = tuple(sorted(set(source_loader_ranks) | set(topology.constructor_ranks)))
    if planner_rank not in data_plane_ranks:
        raise ValueError(f"planner_rank {planner_rank} must belong to data-plane ranks {data_plane_ranks}.")
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed:
        if len(topology.rank_list) != 1:
            raise ValueError("torch.distributed must be initialized for a mesh containing more than one rank.")
        return DataGroups(data_plane_ranks, None, None, planner_rank, False)

    world_ranks = tuple(range(dist.get_world_size()))
    if set(topology.rank_list) != set(world_ranks):
        raise ValueError(
            "The distributed data root mesh must contain every torch.distributed world rank so all ranks create "
            "service groups in the same order."
        )
    if dist.get_rank() != topology.global_rank:
        raise ValueError("Topology global_rank does not match torch.distributed rank.")
    if not isinstance(cpu_backend, str) or not cpu_backend:
        raise ValueError("cpu_backend must be a non-empty backend name.")

    data_plane_group = None
    if len(data_plane_ranks) > 1:
        created = dist.new_group(ranks=list(data_plane_ranks), backend=cpu_backend)
        if topology.global_rank in data_plane_ranks:
            data_plane_group = created

    model_parallel_group = None
    for rank_group in topology.model_parallel_rank_groups:
        if len(rank_group) == 1:
            continue
        created = dist.new_group(ranks=list(rank_group), backend=cpu_backend)
        if topology.global_rank in rank_group:
            model_parallel_group = created
    return DataGroups(data_plane_ranks, data_plane_group, model_parallel_group, planner_rank, True)


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


class DataPlaneTransport:
    """Metadata/control collectives and variable-byte sample all-to-all."""

    def __init__(self, groups: DataGroups, global_rank: int) -> None:
        """Store service-group membership."""
        self._ranks = groups.data_plane_ranks
        self._group = groups.data_plane_group
        self._planner_rank = groups.planner_rank
        self._distributed = groups.distributed
        self._global_rank = global_rank
        self._is_member = global_rank in self._ranks

    @property
    def is_member(self) -> bool:
        """Return whether this rank participates in Source/Planner communication."""
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
        """Gather small control objects on every data-plane rank."""
        self._require_member()
        if len(self._ranks) == 1:
            return (value,)
        gathered = [None] * len(self._ranks)
        dist.all_gather_object(gathered, value, group=self._group)
        return tuple(gathered)

    def gather_object_to_planner(self, value: Any) -> tuple[Any, ...] | None:
        """Gather Source metadata only on the Planner rank."""
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
            group=self._group,
        )
        return tuple(gathered) if gathered is not None else None

    def broadcast_from_planner(self, value: Any) -> Any:
        """Broadcast one control object from the configured Planner."""
        self._require_member()
        if len(self._ranks) == 1:
            if self._global_rank != self._planner_rank:
                raise ValueError("A singleton data plane must contain the Planner rank.")
            return value
        payload = [value if self._global_rank == self._planner_rank else None]
        dist.broadcast_object_list(payload, src=self._planner_rank, group=self._group)
        return payload[0]

    def synchronize_error(self, error: str | None) -> str | None:
        """Return the first rank-ordered error observed by the data plane."""
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
            step: int,
            stopped: bool,
            model_group_error: str | None,
    ) -> str | None:
        """Validate iterator state across all model groups before reading."""
        statuses = self.all_gather_object((self._global_rank, step, stopped, model_group_error))
        normalized = []
        for status in statuses:
            if not isinstance(status, tuple) or len(status) != 4:
                return "Data-plane rank contributed an invalid iterator-state status."
            rank, rank_step, rank_stopped, error = status
            if not isinstance(rank, int) or isinstance(rank, bool) or rank not in self._ranks:
                return "Data-plane rank contributed an invalid iterator-state rank."
            if not isinstance(rank_step, int) or isinstance(rank_step, bool) or rank_step < 0:
                return f"Data-plane rank {rank} contributed an invalid iterator step."
            if not isinstance(rank_stopped, bool):
                return f"Data-plane rank {rank} contributed an invalid stopped state."
            if error is not None and not isinstance(error, str):
                return f"Data-plane rank {rank} contributed a non-string model-group error."
            normalized.append((rank, rank_step, rank_stopped, error))
        contributed_ranks = [status[0] for status in normalized]
        if len(contributed_ranks) != len(set(contributed_ranks)) or set(contributed_ranks) != set(self._ranks):
            return f"Data-plane iterator states have invalid rank coverage {contributed_ranks}."
        errors = sorted((rank, error) for rank, _, _, error in normalized if error is not None)
        if errors:
            return errors[0][1]
        states = {(rank_step, rank_stopped) for _, rank_step, rank_stopped, _ in normalized}
        if len(states) != 1:
            details = ", ".join(
                f"rank {rank}=(step={rank_step}, stopped={rank_stopped})"
                for rank, rank_step, rank_stopped, _ in normalized
            )
            return f"Data-plane ranks have inconsistent iterator state: {details}."
        return None

    def prepare_exchange(
            self,
            outgoing: Mapping[int, Sequence[tuple[SampleKey, Any]]],
    ) -> PreparedPayloadExchange:
        """Serialize and allocate the send buffer before collective entry."""
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
        local_index = self._ranks.index(self._global_rank)
        return PreparedPayloadExchange(
            input_splits=input_splits,
            send_storage=send_storage,
            send_tensor=send_tensor,
            local_segment=segments[local_index],
        )

    def exchange_prepared(self, prepared: PreparedPayloadExchange) -> dict[SampleKey, Any]:
        """Exchange framed sample payloads with variable-split CPU all-to-all."""
        self._require_member()
        if not isinstance(prepared, PreparedPayloadExchange) or len(prepared.input_splits) != len(self._ranks):
            raise ValueError(f"Expected a prepared exchange for {len(self._ranks)} data-plane ranks.")
        if len(self._ranks) == 1:
            return dict(_decode_payload_segment(prepared.local_segment))

        input_splits = list(prepared.input_splits)
        size_input = torch.tensor(input_splits, dtype=torch.int64)
        size_output = torch.empty_like(size_input)
        size_work = dist.all_to_all_single(size_output, size_input, group=self._group, async_op=True)
        if size_work is not None:
            size_work.wait()
        output_splits = [int(size) for size in size_output.tolist()]
        allocation_error = None
        received_tensor = None
        try:
            if len(output_splits) != len(self._ranks) or any(size < 0 for size in output_splits):
                raise ValueError(f"Sample all-to-all returned invalid payload sizes {output_splits}.")
            received_tensor = torch.empty((sum(output_splits),), dtype=torch.uint8)
        except Exception as exc:
            allocation_error = f"Payload receive allocation failed: {type(exc).__name__}: {exc}"
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
            group=self._group,
            async_op=True,
        )
        if data_work is not None:
            data_work.wait()
        received_bytes = received_tensor.numpy().tobytes()
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

    def _require_member(self) -> None:
        if not self._is_member:
            raise ValueError(f"Global rank {self._global_rank} is not a data-plane member.")
        if len(self._ranks) > 1 and (not self._distributed or self._group is None):
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

    def synchronize_iterator_state(self, *, step: int, stopped: bool) -> str | None:
        """Return a shared error when peers restored different iterator state."""
        if len(self._ranks) == 1:
            return None
        if not self._distributed or self._group is None:
            return "Multi-rank model delivery requires an initialized process group."
        gathered = [None] * len(self._ranks)
        dist.all_gather_object(gathered, (self._global_rank, step, stopped), group=self._group)
        normalized = []
        for status in gathered:
            if not isinstance(status, tuple) or len(status) != 3:
                return "A model-parallel peer contributed an invalid iterator-state status."
            rank, rank_step, rank_stopped = status
            if rank not in self._ranks:
                return "A model-parallel peer contributed an invalid iterator-state rank."
            if not isinstance(rank_step, int) or isinstance(rank_step, bool) or rank_step < 0:
                return f"Model-parallel rank {rank} contributed an invalid iterator step."
            if not isinstance(rank_stopped, bool):
                return f"Model-parallel rank {rank} contributed an invalid stopped state."
            normalized.append((rank, rank_step, rank_stopped))
        contributed_ranks = [status[0] for status in normalized]
        if len(contributed_ranks) != len(set(contributed_ranks)) or set(contributed_ranks) != set(self._ranks):
            return f"Model-parallel iterator states have invalid rank coverage {contributed_ranks}."
        states = {(rank_step, rank_stopped) for _, rank_step, rank_stopped in normalized}
        if len(states) != 1:
            details = ", ".join(
                f"rank {rank}=(step={rank_step}, stopped={rank_stopped})"
                for rank, rank_step, rank_stopped in normalized
            )
            return f"Model-parallel ranks have inconsistent iterator state: {details}."
        return None

    def broadcast(self, batch: ConstructedBatch | None) -> ConstructedBatch:
        """Return the constructor's batch envelope on every model-parallel peer."""
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
