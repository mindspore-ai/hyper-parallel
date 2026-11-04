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
"""Infer actual equivalent device replicas from parameter layouts."""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch.distributed as dist

from hyper_parallel.core.fault_tolerance.groups import GroupRegistry
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS, get_created_group


def device_replica_ranks(parameter: Any, rank: int) -> set[int]:
    """Vary replicated mesh axes only; TP/EP shard coordinates must stay fixed."""
    mesh = getattr(parameter, "device_mesh", None)
    placements = getattr(parameter, "placements", None)
    if mesh is None or placements is None:
        return {rank}
    shape = tuple(mesh.mesh_shape)
    ranks = tuple(int(value) for value in mesh.rank_list)
    own = tuple(np.unravel_index(ranks.index(rank), shape))
    result = set()
    for index, coordinate in enumerate(np.ndindex(shape)):
        if all(placement.is_replicate() or coordinate[axis] == own[axis]
               for axis, placement in enumerate(placements)):
            result.add(ranks[index])
    return result


def model_replica_candidates(state: Any, rank: int, world: int) -> tuple[int, ...]:
    """Intersect declared device-shard equivalence across every model parameter."""
    candidates = set(range(world))
    for parameter in state.parameters.values():
        peers = device_replica_ranks(parameter, rank)
        candidates.intersection_update(peers)
    return tuple(sorted(candidates))


@dataclass(frozen=True)
class ReplicaTopology:
    """A partition of equivalent full local device states, never arbitrary backup pairs."""

    replica_groups: tuple[tuple[int, ...], ...]

    @classmethod
    def from_profiles(cls, profiles: list[dict[str, Any]]) -> "ReplicaTopology":
        """Intersect model equivalence with actual optimizer-state ownership."""
        groups = set()
        for rank, profile in enumerate(profiles):
            peers = tuple(profile["candidates"])
            if len(peers) < 2 or rank not in peers:
                raise ValueError(f"rank {rank} has no equivalent device replica; unique TP/EP shards cannot recover "
                                 "and worker recovery cannot be enabled")
            for peer in peers:
                if not 0 <= peer < len(profiles):
                    raise ValueError("replica rank is outside the training world")
                other = profiles[peer]
                if tuple(other["candidates"]) != peers or other["schema"] != profile["schema"]:
                    raise ValueError("device replica layout or optimizer ownership is inconsistent")
            complete = tuple(peer for peer in peers
                             if profiles[peer]["optimizer_ownership"] == profile["optimizer_ownership"])
            if len(complete) < 2:
                raise ValueError(f"rank {rank} has no complete device replica for model and optimizer state; "
                                 f"worker recovery cannot be enabled (model replicas={peers}). "
                                 "Check Muon hsdp_replica_count and its actual state-owner assignments.")
            groups.add(complete)
        if len({len(group) for group in groups}) != 1:
            raise ValueError("MindIO recovery requires uniform replica counts across these device domains")
        return cls(tuple(sorted(groups)))

    @classmethod
    def discover(cls, state: Any, rank: int, world: int, reboot: bool) -> "ReplicaTopology":
        """Derive equivalent owners locally; validate original workers once without caching.

        Args:
            state: Bound model and optimizer state with the local ownership plan.
            rank: Current global worker rank.
            world: Number of workers in the training world.
            reboot: Whether this replacement must avoid global descriptor exchange.

        Returns:
            Complete local-state replicas containing this worker.
        """
        candidates = model_replica_candidates(state, rank, world)
        if len(candidates) < 2:
            raise ValueError(f"rank {rank} has no equivalent device replica; worker recovery cannot be enabled")
        ownership = {peer: state.replica_ownership(peer) for peer in candidates}
        if ownership.get(rank) != state.optimizer_ownership:
            raise ValueError("local Muon owner inference differs from its actual optimizer assignment")
        members = tuple(peer for peer in candidates if ownership[peer] == state.optimizer_ownership)
        if len(members) < 2:
            raise ValueError(f"rank {rank} has no complete device replica for model and optimizer state; "
                             "check Muon hsdp_replica_count and its actual state-owner assignments")
        if not reboot:
            own = {"candidates": candidates, "schema": state.schema,
                   "optimizer_ownership": state.optimizer_ownership}
            profiles = [None] * world
            dist.all_gather_object(profiles, own)
            if cls.from_profiles(profiles).members(rank) != members:
                raise ValueError("derived optimizer replicas disagree with original workers")
        return cls((members,))

    def members(self, rank: int) -> tuple[int, ...]:
        """Return the complete equivalent device-replica domain for this rank."""
        for group in self.replica_groups:
            if rank in group:
                return group
        raise ValueError(f"rank {rank} is outside the device replica topology")

    def get_metadata_group(self, rank: int, registry: GroupRegistry) -> Any:
        """Reuse an exact HCCL replica group or construct a lazy group-local handle.

        Args:
            rank: Global rank whose complete replicas exchange metadata.
            registry: Recovery catalog that preserves handles across communication rebuild.

        Returns:
            An existing or newly registered HCCL group with exactly the replica members.
        """
        ranks = tuple(sorted(self.members(rank)))
        group = get_created_group(ranks)
        if group is None:
            if ranks == tuple(range(dist.get_world_size())):
                group = dist.group.WORLD
            else:
                if os.getenv("TORCH_DIST_INIT_BARRIER", "0") != "0":
                    raise ValueError("metadata group bootstrap requires TORCH_DIST_INIT_BARRIER=0")
                if getattr(dist.group.WORLD, "bound_device_id", None) is not None:
                    raise ValueError("metadata group bootstrap requires a default group without bound device_id")
                # Hashed names depend on members, not the creation count of other
                # replica domains. No device_id keeps replacement HCCL setup lazy.
                group = dist.new_group(ranks=list(ranks), backend="hccl", use_local_synchronization=True,
                                       group_desc="recovery_metadata")
        if str(dist.get_backend(group)) != "hccl":
            raise ValueError("cached metadata group must use HCCL")
        if tuple(sorted(dist.get_process_group_ranks(group))) != ranks:
            raise ValueError("cached metadata group members differ from complete device replicas")
        if group is not dist.group.WORLD:
            registry.register(group, purpose="recovery_metadata")
        EXISTING_COMM_GROUPS[str(ranks)] = group
        return group
