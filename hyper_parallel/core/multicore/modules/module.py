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
"""Execution-resource ownership shared by Torch multicore modules."""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
import torch.distributed as dist

from .. import _automatic


class _ExecutionResourceGroup:
    """Process-owned runtime resources shared by compatible modules."""

    def __init__(
        self,
        identifier: int,
        member_token: int,
        specification: Any,
        compatibility_key: Any,
        scope_key: Any,
    ) -> None:
        """Initialize one unbound process-owned resource group."""
        self.identifier = identifier
        self.members = {member_token}
        self.specification = specification
        self.compatibility_key = compatibility_key
        self.scope_key = scope_key
        self.resources = None
        self.binding = None
        self.shared = False
        self.binding_id = None
        self.closing = False
        self.closed = False
        self.module_type = None


class _MulticoreResourceManager:
    """Track lazy resources while their owning modules are alive."""

    def __init__(self) -> None:
        """Initialize empty process resource state."""
        self._next_group_id = itertools.count()
        self._next_member_token = itertools.count()
        self._next_binding_id = itertools.count()
        self._groups: dict[int, _ExecutionResourceGroup] = {}
        self.runtime_release: Optional[Callable[[], None]] = None
        self._runtime_released = False

    def recycle(self, target: _ExecutionResourceGroup, binding: Any, signature: Any) -> None:
        """Reconcile orphan slots on first binding, never on a local GC decision.

        Args:
            target: New owner requesting resources.
            binding: Local device and dtype binding.
            signature: Rank-independent configuration to compare across peers.
        """
        bound = self._bound_groups(self._groups.values())
        manifest = tuple((group.binding_id, group.resources.lifecycle_signature()) for group in bound)
        states = []
        for group in bound:
            idle = not group.members and not group.closing and group.resources.can_close()
            compatible = (
                (group.compatibility_key, group.binding, group.scope_key, group.shared, group.module_type)
                == (target.compatibility_key, binding, target.scope_key, target.shared, target.module_type)
            )
            states.append((idle, compatible))
        peers = self._exchange((manifest, signature, states))
        if any(peer[:2] != (manifest, signature) for peer in peers):
            raise RuntimeError("multicore resource binding order or specifications differ across ranks")
        reusable = None
        for index, group in enumerate(bound):
            if not all(peer[2][index][0] for peer in peers):
                continue
            if reusable is None and all(peer[2][index][1] for peer in peers):
                reusable = group
            else:
                self._close_group_collectively(group)
        if reusable is not None:
            target.resources, reusable.resources = reusable.resources, None
            target.binding, target.binding_id = reusable.binding, reusable.binding_id
            reusable.closed = True
            self._groups.pop(reusable.identifier)

    def create_group(
        self,
        specification: Any,
        compatibility_key: Any,
        scope_key: Any,
    ) -> tuple[_ExecutionResourceGroup, int]:
        """Create one unbound resource group and its first membership token.

        Args:
            specification: Resource specification supplied by the owning module.
            compatibility_key: Key used to determine whether resources can be shared.
            scope_key: Execution scope that owns the resource group.
        """
        group_id = next(self._next_group_id)
        member_token = next(self._next_member_token)
        group = _ExecutionResourceGroup(
            group_id,
            member_token,
            specification,
            compatibility_key,
            scope_key,
        )
        self._groups[group_id] = group
        return group, member_token

    def add_member(self, group: _ExecutionResourceGroup) -> int:
        """Add a new membership token to an existing group.

        Args:
            group: Existing resource group receiving the member.
        """
        member_token = next(self._next_member_token)
        group.members.add(member_token)
        return member_token

    def retire(self, group_id: int, member_token: int) -> None:
        """Retire membership; native resources wait for coordinated shutdown.

        Args:
            group_id: Identifier of the resource group.
            member_token: Membership token to retire.
        """
        group = self._groups.get(group_id)
        if group is None:
            return
        group.members.discard(member_token)
        if group.members or group.resources is not None:
            return
        group.closed = True
        self._groups.pop(group_id, None)

    @staticmethod
    def _exchange(value: Any) -> list[Any]:
        """Exchange lifecycle metadata on the complete distributed world."""
        return _automatic.exchange(value)

    def close_groups(self, groups: Iterable[_ExecutionResourceGroup] | None = None) -> None:
        """Close all automatic owners, or the last explicit member's group.

        Args:
            groups: Selected groups, or all registered groups for automatic cleanup.
        """
        groups = list(self._groups.values() if groups is None else groups)
        bound = self._bound_groups(groups)
        self._validate_collective_close(bound)
        for group in bound:
            self._close_group_collectively(group)
        for group in groups:
            if group.resources is None:
                group.closed = True
                group.members.clear()
                self._groups.pop(group.identifier, None)
        if self.runtime_release is not None and not any(group.resources is not None
                                                        for group in self._groups.values()):
            _automatic.collective_call(lambda: self._release_runtime(self.runtime_release))
            self.runtime_release = None
            self._runtime_released = False

    def _release_runtime(self, release: Callable[[], None]) -> None:
        """Keep successful local release idempotent until every peer acknowledges it."""
        if not self._runtime_released:
            release()
            self._runtime_released = True

    @staticmethod
    def _bound_groups(groups: Iterable[_ExecutionResourceGroup]) -> list[_ExecutionResourceGroup]:
        """Order native owners by collective binding rather than local construction."""
        return sorted((group for group in groups if group.resources is not None), key=lambda group: group.binding_id)

    def _validate_collective_close(self, bound: list[_ExecutionResourceGroup]) -> None:
        """Require matching idle resources on every rank before release."""
        manifest = tuple((group.binding_id, group.resources.lifecycle_signature()) for group in bound)
        ready = all(group.resources.can_close() for group in bound)
        peers = self._exchange((manifest, ready))
        if any(peer[0] != manifest for peer in peers):
            raise RuntimeError("multicore resource manifests differ across ranks")
        if not all(peer[1] for peer in peers):
            raise RuntimeError("multicore cleanup requires idle workspaces and no pending backward graphs")

    def _close_group_collectively(self, group: _ExecutionResourceGroup) -> None:
        """Keep a failed group reachable and prevent peers from advancing past it."""
        group.closing = True
        _automatic.collective_call(group.resources.close)
        # Commit ownership only after peers also succeed; retries need the same manifest.
        group.resources = None
        group.binding = None
        group.closed = True
        group.members.clear()
        self._groups.pop(group.identifier, None)

    def active_specifications(self, scope_key: Any) -> tuple[Any, ...]:
        """Return one specification per live or native-bound resource group.

        Args:
            scope_key: Execution scope whose resource groups are selected.
        """
        specifications = []
        for group in self._groups.values():
            same_scope = group.scope_key == scope_key
            is_active = bool(group.members) or group.resources is not None
            if same_scope and is_active:
                specifications.append(group.specification)
        return tuple(specifications)


_RESOURCE_MANAGER = _MulticoreResourceManager()


class MulticoreModule(torch.nn.Module):
    """Base class for Torch modules that lazily own multicore resources."""

    def __init__(
        self,
        *,
        resource_specification: Any,
        resource_compatibility_key: Any,
        resource_scope_key: Any,
    ) -> None:
        """Initialize an unbound, process-owned execution-resource group."""
        super().__init__()
        group, member_token = _RESOURCE_MANAGER.create_group(
            resource_specification,
            resource_compatibility_key,
            resource_scope_key,
        )
        self._resource_group = group
        self._resource_member_token = member_token
        self._resource_closed = False
        self._resource_finalizer = weakref.finalize(
            self,
            _RESOURCE_MANAGER.retire,
            group.identifier,
            member_token,
        )

    @classmethod
    def share_execution_resources(cls, modules: Iterable[MulticoreModule]) -> None:
        """Share one synchronous workspace among compatible serial modules.

        Sharing must be configured before any member executes. Modules retain
        independent parameters and optimizer state.

        Args:
            modules: Compatible modules that execute serially.
        """
        members = tuple(modules)
        first_group = cls._validate_shared_members(members)
        if all(
            member._resource_group is first_group  # pylint: disable=protected-access
            for member in members
        ):
            return
        cls._validate_unbound_groups(members, first_group)
        first_group.shared = True
        for member in members[1:]:
            member._move_to_resource_group(first_group)  # pylint: disable=protected-access

    @classmethod
    def _validate_shared_members(
        cls,
        members: tuple[MulticoreModule, ...],
    ) -> _ExecutionResourceGroup:
        """Validate the requested members and return the leading group."""
        if not members:
            raise ValueError("shared multicore execution requires at least one module.")
        if len({id(member) for member in members}) != len(members):
            raise ValueError("shared multicore execution cannot contain duplicate modules.")
        if any(not isinstance(member, cls) for member in members):
            actual = [type(member).__name__ for member in members]
            raise TypeError(
                f"all shared modules must be {cls.__name__} instances, got {actual}."
            )
        if any(member._resource_closed or member._resource_group.closed  # pylint: disable=protected-access
               for member in members):
            raise RuntimeError("closed multicore modules cannot share execution resources.")
        if any(member._resource_group.closing  # pylint: disable=protected-access
               for member in members):
            raise RuntimeError("closing multicore modules cannot share execution resources.")
        concrete_types = {type(member) for member in members}
        if len(concrete_types) != 1:
            raise TypeError("shared multicore execution requires one concrete module type.")
        return members[0]._resource_group  # pylint: disable=protected-access

    @staticmethod
    def _validate_unbound_groups(
        members: tuple[MulticoreModule, ...],
        first_group: _ExecutionResourceGroup,
    ) -> None:
        """Require compatible, single-member groups before merging them."""
        for member in members:
            group = member._resource_group  # pylint: disable=protected-access
            if group.compatibility_key != first_group.compatibility_key:
                raise ValueError(
                    "shared multicore modules must have identical configuration and scope."
                )
            if group.resources is not None or group.binding is not None or len(group.members) != 1:
                raise RuntimeError(
                    "execution resources must be shared before first use or previous grouping."
                )

    def _move_to_resource_group(self, target: _ExecutionResourceGroup) -> None:
        """Move this module to ``target`` without invoking native cleanup."""
        old_group = self._resource_group
        old_token = self._resource_member_token
        self._resource_finalizer.detach()
        _RESOURCE_MANAGER.retire(
            old_group.identifier,
            old_token,
        )
        member_token = _RESOURCE_MANAGER.add_member(target)
        self._resource_group = target
        self._resource_member_token = member_token
        self._resource_finalizer = weakref.finalize(
            self,
            _RESOURCE_MANAGER.retire,
            target.identifier,
            member_token,
        )

    def _get_execution_resources(self, tensor: Any) -> Any:
        """Create or return resources bound to ``tensor`` device and dtype."""
        if self._resource_closed or self._resource_group.closed:
            raise RuntimeError("cannot execute a closed multicore module.")
        if self._resource_group.closing:
            raise RuntimeError("cannot execute a closing multicore module; finish cleanup before reuse.")
        group = self._resource_group
        binding = self._execution_binding(tensor)
        if group.resources is None:
            group.module_type = type(self)
            _RESOURCE_MANAGER.recycle(group, binding, self._resource_signature(tensor))
        if group.resources is None:
            _automatic.register(_RESOURCE_MANAGER.close_groups, self._root_group())
            active_specs = _RESOURCE_MANAGER.active_specifications(group.scope_key)
            group.resources = self._create_execution_resources(
                tensor,
                shared=group.shared,
                active_specifications=active_specs,
            )
            group.binding = binding
            group.binding_id = next(_RESOURCE_MANAGER._next_binding_id)  # pylint: disable=protected-access
            if _RESOURCE_MANAGER.runtime_release is None:
                self._retain_runtime(_RESOURCE_MANAGER)
        elif binding != group.binding:
            raise ValueError(
                "multicore resources are bound to the first input device/dtype "
                f"{group.binding}, got {binding}."
            )
        return group.resources

    def _resource_signature(self, tensor: Any) -> Any:
        """Identify a collective first binding without rank-local device indices."""
        return type(self).__module__, type(self).__qualname__, str(tensor.dtype)

    def _retain_runtime(self, manager: _MulticoreResourceManager) -> None:
        """Optionally pin the runtime across orphan replacement allocations."""

    def _root_group(self) -> Any:
        """Return the communication dependency used by these resources."""
        return dist.group.WORLD

    @staticmethod
    def _execution_binding(tensor: Any) -> tuple[Any, Any]:
        """Return the tensor binding used for resource compatibility."""
        try:
            return tensor.device, tensor.dtype
        except AttributeError as error:
            raise TypeError("multicore modules require a Torch tensor input.") from error

    def _create_execution_resources(
        self,
        tensor: Any,
        *,
        shared: bool,
        active_specifications: tuple[Any, ...],
    ) -> Any:
        """Create resources for the first input binding."""
        raise NotImplementedError

    def close(self) -> None:
        """Release this member, closing shared resources only after the last member.

        Call in the same order on all WORLD ranks after the last backward.
        Recoverable failures retain ownership for retry; partially closed modules
        cannot execute again. Native communication failures may require restart.
        """
        if self._resource_closed:
            return
        group = self._resource_group
        if group.resources is not None and group.members == {self._resource_member_token}:
            _RESOURCE_MANAGER.close_groups((group,))
        elif group.closed:
            # A previous attempt may have freed buffers but failed the final runtime release.
            _RESOURCE_MANAGER.close_groups(())
        else:
            _RESOURCE_MANAGER.retire(group.identifier, self._resource_member_token)
        self._resource_closed = True
        self._resource_finalizer.detach()
