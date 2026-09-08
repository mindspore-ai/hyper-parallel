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
from collections.abc import Iterable
from typing import Any

import torch



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


class _MulticoreResourceManager:
    """Track lazy resources while their owning modules are alive."""

    def __init__(self) -> None:
        """Initialize empty process resource state."""
        self._next_group_id = itertools.count()
        self._next_member_token = itertools.count()
        self._groups: dict[int, _ExecutionResourceGroup] = {}

    def create_group(
        self,
        specification: Any,
        compatibility_key: Any,
        scope_key: Any,
    ) -> tuple[_ExecutionResourceGroup, int]:
        """Create one unbound resource group and its first membership token."""
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
        """Add a new membership token to an existing group."""
        member_token = next(self._next_member_token)
        group.members.add(member_token)
        return member_token

    def retire(
        self,
        group_id: int,
        member_token: int,
        *,
        close_resources: bool,
    ) -> None:
        """Retire one member and optionally close the last native owner."""
        group = self._groups.get(group_id)
        if group is None:
            return
        group.members.discard(member_token)
        if group.members:
            return
        resources = group.resources
        if resources is None:
            self._groups.pop(group_id, None)
            return
        if not close_resources:
            return
        group.resources = None
        group.binding = None
        self._groups.pop(group_id, None)
        resources.close()

    def active_specifications(self, scope_key: Any) -> tuple[Any, ...]:
        """Return one specification per live or native-bound resource group."""
        return tuple(
            group.specification
            for group in self._groups.values()
            if group.scope_key == scope_key
            and (group.members or group.resources is not None)
        )


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
            close_resources=False,
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
        if any(member._resource_closed for member in members):  # pylint: disable=protected-access
            raise RuntimeError("closed multicore modules cannot share execution resources.")
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
            close_resources=False,
        )
        member_token = _RESOURCE_MANAGER.add_member(target)
        self._resource_group = target
        self._resource_member_token = member_token
        self._resource_finalizer = weakref.finalize(
            self,
            _RESOURCE_MANAGER.retire,
            target.identifier,
            member_token,
            close_resources=False,
        )

    def _get_execution_resources(self, tensor: Any) -> Any:
        """Create or return resources bound to ``tensor`` device and dtype."""
        if self._resource_closed:
            raise RuntimeError("cannot execute a closed multicore module.")
        group = self._resource_group
        binding = self._execution_binding(tensor)
        if group.resources is None:
            active_specs = _RESOURCE_MANAGER.active_specifications(group.scope_key)
            group.resources = self._create_execution_resources(
                tensor,
                shared=group.shared,
                active_specifications=active_specs,
            )
            group.binding = binding
        elif binding != group.binding:
            raise ValueError(
                "multicore resources are bound to the first input device/dtype "
                f"{group.binding}, got {binding}."
            )
        return group.resources

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
        """Release this module's membership and last-owned native resources."""
        if self._resource_closed:
            return
        self._resource_closed = True
        self._resource_finalizer.detach()
        _RESOURCE_MANAGER.retire(
            self._resource_group.identifier,
            self._resource_member_token,
            close_resources=True,
        )
