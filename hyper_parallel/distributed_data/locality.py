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
"""Create independent node-local data exchange groups."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import os
import socket
from typing import Any

import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import DataGroups, all_gather_control_object


def _node_rank_groups(
        topology: DataTopology,
        node_ids: dict[int, str],
) -> tuple[tuple[int, ...], ...]:
    buckets: dict[str, list[int]] = {}
    for rank in topology.rank_list:
        buckets.setdefault(node_ids[rank], []).append(rank)
    return tuple(sorted(tuple(sorted(ranks)) for ranks in buckets.values()))


def _resolve_local_topology(
        mesh: Any,
        rank: int,
        world_size: int,
        dp_dim_names: tuple[str, ...] | None,
) -> DataTopology:
    """Resolve the WORLD-covering topology for independent DP readers."""
    topology = DataTopology.from_mesh(mesh, global_rank=rank, dp_dim_names=dp_dim_names)
    if set(topology.rank_list) != set(range(world_size)):
        raise ValueError("Local balancing requires a root mesh containing every WORLD rank.")
    if any(len(group) != 1 for group in topology.model_parallel_rank_groups):
        raise ValueError("Local balancing currently requires CP/TP/PP/Ulysses and other model parallel sizes = 1.")
    return topology


def _validate_startup_statuses(
        statuses: list[Any],
        identity: Any,
) -> None:
    """Check the gathered startup results before creating node groups."""
    for entry in statuses:
        if entry[1] is not None:
            raise ValueError(f"Local balancing build failed on rank {entry[0]}: {entry[1]}")
    if any(entry[2] != identity for entry in statuses):
        raise ValueError("Local balancing configuration or root mesh differs across WORLD ranks.")


def _validate_communication_config(backend: str, device: Any, distributed: bool) -> None:
    """Validate the node-local collective backend and its rank-local device."""
    if backend not in ("gloo", "hccl"):
        raise ValueError("communication_backend must be 'gloo' or 'hccl'.")
    if backend != "hccl":
        return
    if distributed and (device is None or getattr(device, "type", None) != "npu"):
        raise ValueError("communication_backend='hccl' requires an NPU communication_device.")


def _gather_startup_statuses(
        status: Any,
        *,
        distributed: bool,
        backend: str,
        device: Any,
) -> list[Any]:
    """Gather startup state on an explicit Gloo group or HCCL WORLD."""
    if not distributed:
        return [status]
    startup_group = None
    if backend == "gloo":
        startup_group = dist.new_group(ranks=list(range(dist.get_world_size())), backend="gloo")
    return list(all_gather_control_object(status, group=startup_group, device=device, backend=backend))


def _create_locality_groups(
        mesh: Any,
        *,
        dp_dim_names: tuple[str, ...] | None = None,
        build_identity: Any = None,
        local_error: str | None = None,
        communication_backend: str = "hccl",
        communication_device: Any = None,
) -> tuple[DataTopology, DataGroups]:
    """Resolve locality once and create all process groups in WORLD rank order.

    Args:
        mesh: Named root mesh covering WORLD, with no nontrivial model parallelism.
        dp_dim_names: Root mesh data-parallel dimensions.
        build_identity: Rank-independent caller configuration checked during startup.
        local_error: Caller preflight error, synchronized before subgroup creation.

    Returns:
        Training topology and this rank's independent data exchange groups.

    Note:
        WORLD communication is used only during startup. No training mesh or
        FSDP process group is modified. All ranks must call this factory together.
    """
    distributed = dist.is_available() and dist.is_initialized()
    _validate_communication_config(communication_backend, communication_device, distributed)
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    topology = None
    identity = None
    selected_node = None
    try:
        if local_error is not None:
            raise ValueError(local_error)
        topology = _resolve_local_topology(mesh, rank, world_size, dp_dim_names)
        selected_node = os.environ.get("GROUP_RANK") or socket.gethostname()
        identity = (topology.fingerprint, build_identity)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    status = (rank, local_error, identity, selected_node)
    statuses = _gather_startup_statuses(
        status,
        distributed=distributed,
        backend=communication_backend,
        device=communication_device,
    )
    _validate_startup_statuses(statuses, identity)
    rank_groups = _node_rank_groups(topology, {entry[0]: entry[3] for entry in statuses})
    own_groups = None
    for ranks in rank_groups:
        control_group = None
        payload_group = None
        if distributed and len(ranks) > 1:
            control_group = dist.new_group(ranks=list(ranks), backend=communication_backend)
            payload_group = control_group
            if communication_backend == "hccl":
                payload_group = dist.new_group(ranks=list(ranks), backend=communication_backend)
        if rank in ranks:
            own_groups = DataGroups(ranks, control_group, payload_group, None, min(ranks), distributed)
    return topology, own_groups
