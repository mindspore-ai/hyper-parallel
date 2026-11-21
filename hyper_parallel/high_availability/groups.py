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
"""Track process groups through scoped Torch hooks during worker recovery."""

import inspect
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

_ACTIVE_REGISTRY = ContextVar("hp_recovery_groups", default=None)
_NEW_GROUP_SIGNATURE = inspect.signature(dist.new_group)


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


def recovery_optimizer_subgroup(parent: Any, sub_size: int) -> Any:
    """Split recorded mesh groups without a world collective during worker reboot.

    Return None outside recovery so normal optimizer construction is unchanged.
    All ranks have the canonical mesh-group catalog, including nonmember groups.

    Args:
        parent: Recorded parent process group to split.
        sub_size: Number of ranks in each optimizer subgroup.

    Returns:
        The local optimizer subgroup, or None when recovery is inactive.
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
                group = dist.new_group(
                    ranks=list(members),
                    backend="hccl",
                    group_desc="recovery_optimizer",
                )
                groups[members] = group
        registry.optimizer_subgroups[key] = groups
    rank = dist.get_rank()
    start = (parent_ranks.index(rank) // sub_size) * sub_size
    return registry.optimizer_subgroups[key][parent_ranks[start:start + sub_size]]


class GroupRegistry:
    """Preserve process group identities retained by mesh, buckets and optimizer.

    This context patches Torch's public group factories process-wide during
    construction and training. Previously bound aliases and private factories
    require explicit ``register`` calls. Only fixed-topology HCCL is supported.
    """

    def __init__(self) -> None:
        """Start a fresh deterministic construction catalog."""
        self.groups: list[GroupDescriptor] = []
        self._token = None
        self._original_new_group = None
        self._original_c10d_new_group = None
        self.reboot = False
        self.sealed = False
        self.optimizer_subgroups: dict[Any, Any] = {}

    def __enter__(self) -> "GroupRegistry":
        """Record process groups created in the current execution context."""
        if (_ACTIVE_REGISTRY.get() is not None
                or isinstance(getattr(dist.new_group, "__self__", None), GroupRegistry)):
            raise RuntimeError("nested recovery group registries are not supported")
        self._original_new_group = dist.new_group
        self._original_c10d_new_group = dist.distributed_c10d.new_group
        dist.new_group = self.new_group
        dist.distributed_c10d.new_group = self.new_group
        self._token = _ACTIVE_REGISTRY.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        """Restore the caller's construction context."""
        dist.new_group = self._original_new_group
        dist.distributed_c10d.new_group = self._original_c10d_new_group
        self._original_new_group = None
        self._original_c10d_new_group = None
        _ACTIVE_REGISTRY.reset(self._token)

    def new_group(self, *args: Any, **kwargs: Any) -> Any:
        """Validate and record native group creation for this recovery registry.

        Args:
            *args: Native positional group arguments.
            **kwargs: Native keyword group arguments.

        Returns:
            The native process group or nonmember sentinel.
        """
        options = _NEW_GROUP_SIGNATURE.bind(*args, **kwargs).arguments
        if options.get("use_local_synchronization", False) or options.get("device_id") is not None:
            raise ValueError("recovery groups require lazy init and no local synchronization")
        if getattr(dist.group.WORLD, "bound_device_id", None) is not None:
            raise ValueError("recovery groups require a default group without bound device_id")
        if self.sealed:
            raise RuntimeError("materialize every recovery process group during bootstrap, before attach")
        if options.get("pg_options") is not None:
            raise ValueError("custom HCCL process-group options are not supported by worker recovery")
        backend = str(options.get("backend") or dist.get_backend())
        if backend != "hccl" or os.getenv("TORCH_DIST_INIT_BARRIER", "0") != "0":
            raise ValueError("worker reboot construction requires lazy HCCL and TORCH_DIST_INIT_BARRIER=0")
        group = self.call_original_new_group(*args, **kwargs)
        ranks = tuple(sorted(options.get("ranks") or range(dist.get_world_size())))
        self.groups.append(GroupDescriptor(ranks, backend, options.get("group_desc") or "", group))
        return group

    def call_original_new_group(self, *args: Any, **kwargs: Any) -> Any:
        """Call the saved group factory without reentering the recovery hook.

        Args:
            *args: Native positional group arguments.
            **kwargs: Native keyword group arguments.

        Returns:
            A native group requiring explicit registration or caller-owned teardown.
        """
        create = self._original_new_group if self._original_new_group is not None else dist.new_group
        return create(*args, **kwargs)

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
        seen = {id(dist.group.WORLD)}
        for descriptor in self.groups:
            if rank in descriptor.ranks and id(descriptor.group) not in seen:
                reinit(group=descriptor.group, rebuild_link=True)
                seen.add(id(descriptor.group))
