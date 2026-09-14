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
"""Create independent node-local or HSDP-shard-local data exchange groups."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import os
import socket
from itertools import product
from typing import Any

import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.topology import DataTopology
# Reuse the data plane's backend/device checks; its public group factory assumes
# one WORLD-wide data domain and cannot create independent locality domains.
from hyper_parallel.distributed_data.transport import (
    DataGroups,
    _resolve_payload_backend,
    _validate_group_backends,
)


def _scope_rank_groups(
        topology: DataTopology,
        scope: str,
        node_ids: dict[int, str],
) -> tuple[tuple[int, ...], ...]:
    buckets: dict[Any, list[int]] = {}
    shard_index = topology.mesh_dim_names.index("dp_shard") if scope == "hsdp_shard" else None
    coordinates = product(*(range(size) for size in topology.mesh_shape))
    for rank, coordinate in zip(topology.rank_list, coordinates, strict=True):
        if scope == "node":
            key = node_ids[rank]
        else:
            key = tuple(value for index, value in enumerate(coordinate) if index != shard_index)
        buckets.setdefault(key, []).append(rank)
    return tuple(sorted(tuple(sorted(ranks)) for ranks in buckets.values()))


def create_locality_groups(
        mesh: Any,
        *,
        balancing_scope: str = "node",
        node_id: str | int | None = None,
        dp_dim_names: tuple[str, ...] | None = None,
        cpu_backend: str = "gloo",
        payload_backend: str | None = None,
        communication_device: Any = None,
        build_identity: Any = None,
        local_error: str | None = None,
) -> tuple[DataTopology, DataGroups]:
    """Resolve locality once and create all process groups in WORLD rank order.

    Args:
        mesh: Named root mesh covering WORLD, with no nontrivial model parallelism.
        balancing_scope: ``node`` or ``hsdp_shard``. The latter varies only the
            actual mesh's ``dp_shard`` coordinate, not contiguous global ranks.
        node_id: Explicit node identity, otherwise launcher ``GROUP_RANK`` or hostname.
        dp_dim_names: Root mesh data-parallel dimensions.
        cpu_backend: CPU object-collective backend, normally Gloo.
        payload_backend: Optional data plane backend, normally HCCL/NCCL on accelerators.
        communication_device: Rank-local device used by the existing payload codec.
        build_identity: Rank-independent caller configuration checked during startup.
        local_error: Caller preflight error, synchronized before subgroup creation.

    Returns:
        Training topology and this rank's independent data exchange groups.

    Note:
        WORLD communication is used only during startup. No training mesh or
        FSDP process group is modified. All ranks must call this factory together.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    topology = None
    identity = None
    selected_node = None
    effective_payload = None
    reuse_control = False
    try:
        if local_error is not None:
            raise ValueError(local_error)
        if balancing_scope not in ("node", "hsdp_shard"):
            raise ValueError("balancing_scope must be 'node' or 'hsdp_shard'.")
        topology = DataTopology.from_mesh(mesh, global_rank=rank, dp_dim_names=dp_dim_names)
        if set(topology.rank_list) != set(range(world_size)):
            raise ValueError("Local balancing requires a root mesh containing every WORLD rank.")
        if any(len(group) != 1 for group in topology.model_parallel_rank_groups):
            raise ValueError("Local balancing currently requires CP/TP/PP/Ulysses and other model parallel sizes = 1.")
        if balancing_scope == "hsdp_shard" and "dp_shard" not in topology.dp_dim_names:
            raise ValueError("hsdp_shard balancing requires a named data-parallel 'dp_shard' mesh dimension.")
        selected_node = node_id if node_id is not None else os.environ.get("GROUP_RANK", socket.gethostname())
        if type(selected_node) not in (str, int) or not str(selected_node).strip():
            raise ValueError("node_id must be a non-empty string or integer.")
        selected_node = str(selected_node)
        _validate_group_backends(cpu_backend, payload_backend, True)
        effective_payload, reuse_control = _resolve_payload_backend(
            data_plane_ranks=topology.rank_list,
            cpu_backend=cpu_backend,
            payload_backend=payload_backend,
            communication_device=communication_device,
            enable_payload_exchange=True,
        )
        identity = (
            topology.fingerprint, balancing_scope, cpu_backend, effective_payload, reuse_control, build_identity,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    status = (rank, local_error, identity, selected_node)
    statuses = [status]
    if distributed:
        statuses = [None] * world_size
        dist.all_gather_object(statuses, status)
    for expected_rank, entry in enumerate(statuses):
        if not isinstance(entry, tuple) or len(entry) != 4 or entry[0] != expected_rank:
            raise ValueError("Local balancing received an invalid WORLD startup status.")
        if entry[1] is not None:
            raise ValueError(f"Local balancing build failed on rank {entry[0]}: {entry[1]}")
    if topology is None or any(entry[2] != identity for entry in statuses):
        raise ValueError("Local balancing configuration or root mesh differs across WORLD ranks.")
    rank_groups = _scope_rank_groups(topology, balancing_scope, {entry[0]: entry[3] for entry in statuses})
    own_groups = None
    for ranks in rank_groups:
        control_group = None
        payload_group = None
        if distributed and len(ranks) > 1:
            control_group = dist.new_group(ranks=list(ranks), backend=cpu_backend)
            payload_group = control_group if reuse_control else dist.new_group(
                ranks=list(ranks), backend=effective_payload,
            )
        if rank in ranks:
            own_groups = DataGroups(ranks, control_group, payload_group, None, min(ranks), distributed)
    if own_groups is None:
        raise ValueError(f"Local balancing did not assign rank {rank} to an exchange domain.")
    return topology, own_groups


__all__ = ["create_locality_groups"]
