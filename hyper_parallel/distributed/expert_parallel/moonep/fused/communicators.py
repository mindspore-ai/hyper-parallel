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
"""Exclusive reusable W/G communicators, owned by the Torch distributed session.

Runtime close drains and releases its lease; WORLD teardown destroys the process
groups through Torch's normal registry. Keeping a bounded set per EP membership
avoids connection churn and reuse of stale subgroup Store keys after close.
No expert weights, token windows or activation tensors enter this cache.
"""

from dataclasses import dataclass
from datetime import timedelta
import inspect
from threading import RLock
from typing import Any

import torch.distributed as dist


_lock = RLock()
_world: Any = None
_bundles: dict[tuple[int, ...], list["CommunicatorLease"]] = {}


@dataclass
class CommunicatorLease:
    """One exclusive slot, reusable after all of its streams have drained."""

    members: tuple[int, ...]
    slot: int
    active: bool = False
    groups: tuple[Any, ...] = ()

    def initialize(self) -> tuple[Any, ...]:
        """Create at most two payload and three host-control groups for this slot.

        All EP members must reserve and initialize matching slots in matching
        order. A construction failure leaves the slot active: partially created
        groups require whole-job termination, not local retry.
        """
        if not self.active:
            raise RuntimeError("communicator slot must be reserved before initialization")
        if not self.groups:
            options = {"ranks": self.members, "use_local_synchronization": True}
            if "sort_ranks" in inspect.signature(dist.new_group).parameters:
                options["sort_ranks"] = False
            payload = tuple(dist.new_group(backend="hccl", **options) for _ in range(2))
            control = tuple(dist.new_group(backend="gloo", timeout=timedelta(seconds=60), **options)
                            for _ in range(3))
            self.groups = payload + control
            if any(tuple(dist.get_process_group_ranks(group)) != self.members for group in self.groups):
                raise RuntimeError("auxiliary process-group rank order differs from the EP root")
        return self.groups

    def release(self) -> None:
        """Return an exclusive slot after its runtime drains all data users."""
        with _lock:
            if not self.active:
                raise RuntimeError("communicator lease was already released")
            self.active = False


def reserve_communicators(members: tuple[int, ...]) -> CommunicatorLease:
    """Reserve a session-scoped bundle without allocating communication resources.

    The slot number participates in the runtime's existing collective manifest
    admission before any new group is constructed. Simultaneously active
    runtimes get distinct slots; sequential runtimes reuse an idle slot.
    """
    global _world  # pylint: disable=global-statement
    with _lock:
        if _world is not dist.group.WORLD:
            # Torch has already destroyed the previous session's registered PGs.
            _bundles.clear()
            _world = dist.group.WORLD
        entries = _bundles.setdefault(members, [])
        lease = next((entry for entry in entries if not entry.active), None)
        if lease is None:
            lease = CommunicatorLease(members, len(entries))
            entries.append(lease)
        lease.active = True
        return lease
