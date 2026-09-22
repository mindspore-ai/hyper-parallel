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
"""Version-isolated adapters for cleanup before Torch communication teardown."""

import atexit
import functools
import logging
from collections.abc import Callable
from typing import Any, Optional

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

_LOGGER = logging.getLogger(__name__)


def exchange(value: Any) -> list[Any]:
    """Exchange cold-path lifecycle metadata across the complete world.

    Args:
        value: Picklable local lifecycle state.
    """
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def collective_call(callback: Callable[[], None]) -> None:
    """Keep peers at the same teardown stage when a local operation raises.

    Args:
        callback: Teardown stage entered by every rank in the same order.
    """
    failure = None
    try:
        callback()
    # Peers must see ordinary local errors before entering the next collective.
    except Exception as error:  # pylint: disable=broad-exception-caught
        failure = error
    failures = exchange(None if failure is None else f"{type(failure).__name__}: {failure}")
    if any(error is not None for error in failures):
        raise RuntimeError(f"multicore teardown stage failed: {failures}") from failure


class _Cleanup:
    """Own a single exit hook and dependencies for the current runtime generation."""

    def __init__(self) -> None:
        """Start without native owners or patched framework entry points."""
        self.callback: Optional[Callable[[], None]] = None
        self.dependencies: set[Any] = set()
        self.installed = False

    def close(self) -> None:
        """Keep both the callback and communication available if release fails."""
        if self.callback is not None:
            self.callback()
            self.callback = None
            self.dependencies.clear()

    def at_exit(self) -> None:
        """Best-effort cleanup before torch_npu closes its device runtime."""
        try:
            self.close()
        # Exit must continue through the framework's own device cleanup on failure.
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.exception("Automatic multicore cleanup failed during process exit")

    def install(self) -> None:
        """Intercept the backend boundary, including previously imported destroy aliases."""
        if self.installed:
            return
        if hasattr(c10d.ProcessGroup, "shutdown"):
            owner, name = c10d.ProcessGroup, "shutdown"
        elif hasattr(c10d, "_shutdown_backend"):
            owner, name = c10d, "_shutdown_backend"
        else:
            raise RuntimeError("unsupported Torch process-group lifecycle; no automatic cleanup boundary")
        original = getattr(owner, name)

        @functools.wraps(original)
        def shutdown(group: Any, *args: Any, **kwargs: Any) -> Any:
            """Run cleanup before a dependent communicator is destroyed.

            Args:
                group: Communication dependency being destroyed.
                *args: Original framework arguments.
                **kwargs: Original framework keyword arguments.
            """
            if group in self.dependencies:
                self.close()
            return original(group, *args, **kwargs)

        setattr(owner, name, shutdown)
        # A real NPU input requires torch_npu initialization before this first binding.
        atexit.register(self.at_exit)
        self.installed = True


_CLEANUP = _Cleanup()


def register(callback: Callable[[], None], root_group: Any) -> None:
    """Register native owners without changing the user's process-group API.

    Args:
        callback: Collective release operation; failures must preserve ownership.
        root_group: SHMEM Root dependency, in addition to WORLD metadata exchange.
    """
    _CLEANUP.install()
    _CLEANUP.callback = callback
    _CLEANUP.dependencies.update(group for group in (root_group, dist.group.WORLD) if group is not None)
