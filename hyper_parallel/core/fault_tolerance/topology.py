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

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_store

from hyper_parallel.core.fault_tolerance.groups import new_group

_TOPOLOGY_KEY = "hyper_parallel/device_recovery_topology"


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
        """Exchange descriptors once; replacements read the existing rendezvous store."""
        own = {"candidates": model_replica_candidates(state, rank, world), "schema": state.schema,
               "optimizer_ownership": state.optimizer_ownership}
        store = _get_default_store()
        if reboot:
            profiles = json.loads(store.get(_TOPOLOGY_KEY).decode())
            if (list(own["candidates"]) != profiles[rank]["candidates"]
                    or own["schema"] != profiles[rank]["schema"]
                    or own["optimizer_ownership"] != profiles[rank]["optimizer_ownership"]):
                raise ValueError("replacement model/optimizer layout differs from the original worker")
        else:
            profiles = [None] * world
            dist.all_gather_object(profiles, own)
            cls.from_profiles(profiles)
            store.set(_TOPOLOGY_KEY, json.dumps(profiles))
        return cls.from_profiles(profiles)

    def members(self, rank: int) -> tuple[int, ...]:
        """Return the complete equivalent device-replica domain for this rank."""
        for group in self.replica_groups:
            if rank in group:
                return group
        raise ValueError(f"rank {rank} is outside the device replica topology")

    def create_group(self) -> Any:
        """Create every repair domain in a deterministic order, before attach is sealed."""
        current = None
        rank = dist.get_rank()
        for members in self.replica_groups:
            group = new_group(ranks=list(members), backend="hccl", group_desc="device_recovery")
            if rank in members:
                current = group
        return current
