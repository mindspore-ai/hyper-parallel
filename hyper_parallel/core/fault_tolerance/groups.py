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
"""Track actual training and backup process groups without patching Torch APIs."""

import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

_ACTIVE_REGISTRY = ContextVar("hp_recovery_groups", default=None)


def is_recovery_enabled() -> bool:
    """Return whether the current execution context belongs to a recovery session."""
    return _ACTIVE_REGISTRY.get() is not None


def is_recovery_reboot() -> bool:
    """Return whether the current session belongs to a replacement worker."""
    registry = _ACTIVE_REGISTRY.get()
    return registry is not None and registry.reboot


def is_recovery_bootstrapping() -> bool:
    """Return whether a replacement is still constructing objects before attach."""
    registry = _ACTIVE_REGISTRY.get()
    return registry is not None and registry.reboot and not registry.sealed


@dataclass(frozen=True)
class GroupDescriptor:
    """One creation-ordered process group, including its backend and purpose."""

    ranks: tuple[int, ...]
    backend: str
    purpose: str
    group: Any


def new_group(*args: Any, **kwargs: Any) -> Any:
    """Create a group normally, recording it only inside a recovery bootstrap.

    Recovery construction requires lazy HCCL groups and no store barrier.
    Replacement workers execute the same deterministic construction order;
    they must not run model collectives before the rebuild callback.
    """
    registry = _ACTIVE_REGISTRY.get()
    if registry is None:
        return dist.new_group(*args, **kwargs)
    if args or kwargs.get("use_local_synchronization", False) or kwargs.get("device_id") is not None:
        raise ValueError("recovery groups require named arguments, lazy init and no local synchronization")
    if registry.sealed:
        raise RuntimeError("materialize every recovery process group during bootstrap, before attach")
    if kwargs.get("pg_options") is not None:
        raise ValueError("custom HCCL process-group options are not supported by worker recovery")
    backend = str(kwargs.get("backend") or dist.get_backend())
    if backend != "hccl" or os.getenv("TORCH_DIST_INIT_BARRIER", "0") != "0":
        raise ValueError("worker reboot construction requires lazy HCCL and TORCH_DIST_INIT_BARRIER=0")
    group = dist.new_group(**kwargs)
    ranks = tuple(sorted(kwargs.get("ranks") or range(dist.get_world_size())))
    registry.groups.append(GroupDescriptor(ranks, backend, kwargs.get("group_desc") or "", group))
    return group


def recovery_optimizer_subgroup(parent: Any, sub_size: int) -> Any:
    """Split recorded mesh groups without a world collective during worker reboot.

    Return None outside recovery so normal optimizer construction is unchanged.
    All ranks have the canonical mesh-group catalog, including nonmember groups.
    """
    registry = _ACTIVE_REGISTRY.get()
    if registry is None:
        return None
    parent_ranks = tuple(sorted(dist.get_process_group_ranks(parent)))
    parents = {descriptor.ranks for descriptor in registry.groups
               if len(descriptor.ranks) == len(parent_ranks) and descriptor.purpose != "recovery_backup"}
    if len(parent_ranks) == dist.get_world_size():
        parents.add(tuple(range(dist.get_world_size())))
    if parent_ranks not in parents:
        raise ValueError("register the optimizer's parent mesh groups before recovery subgroup construction")
    key = (tuple(sorted(parents)), sub_size)
    if key not in registry.optimizer_subgroups:
        groups = {}
        for ranks in key[0]:
            if len(ranks) % sub_size:
                raise ValueError("optimizer subgroup size must divide each recorded parent group")
            for start in range(0, len(ranks), sub_size):
                members = ranks[start:start + sub_size]
                group = new_group(ranks=list(members), backend="hccl", group_desc="recovery_optimizer")
                groups[members] = group
        registry.optimizer_subgroups[key] = groups
    rank = dist.get_rank()
    start = (parent_ranks.index(rank) // sub_size) * sub_size
    return registry.optimizer_subgroups[key][parent_ranks[start:start + sub_size]]


class GroupRegistry:
    """Preserve process group identities retained by mesh, buckets and optimizer.

    Use this context around *all* HyperParallel mesh/model construction.
    External process groups must be registered explicitly with ``register``.
    Only fixed-topology HCCL recovery is supported.
    """

    def __init__(self) -> None:
        """Start a fresh deterministic construction catalog."""
        self.groups: list[GroupDescriptor] = []
        self._token = None
        self.generation = 0
        self.reboot = False
        self.sealed = False
        self.optimizer_subgroups: dict[Any, Any] = {}

    def __enter__(self) -> "GroupRegistry":
        """Record process groups created in the current execution context."""
        if _ACTIVE_REGISTRY.get() is not None:
            raise RuntimeError("nested recovery group registries are not supported")
        self._token = _ACTIVE_REGISTRY.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        """Restore the caller's construction context."""
        _ACTIVE_REGISTRY.reset(self._token)

    def register(self, group: Any, purpose: str = "external") -> None:
        """Record a caller-owned group used by the protected model."""
        if self.sealed:
            raise RuntimeError("register process groups before attaching recovery state")
        if any(descriptor.group is group for descriptor in self.groups):
            return
        backend = str(dist.get_backend(group))
        if backend != "hccl":
            raise ValueError("only HCCL process groups support this worker reboot path")
        self.groups.append(GroupDescriptor(tuple(dist.get_process_group_ranks(group)), backend, purpose, group))

    def rebuild(self) -> None:
        """Reinitialize member groups locally; MindIO fences completion before repair."""
        reinit = getattr(dist, "reinit_process_group", None)
        if not callable(reinit):
            raise RuntimeError("torch-npu reinit_process_group is required for process recovery")
        reinit(group=None, rebuild_link=True)
        rank = dist.get_rank()
        seen = set()
        for descriptor in self.groups:
            if rank in descriptor.ranks and id(descriptor.group) not in seen:
                reinit(group=descriptor.group, rebuild_link=True)
                seen.add(id(descriptor.group))
        self.generation += 1
