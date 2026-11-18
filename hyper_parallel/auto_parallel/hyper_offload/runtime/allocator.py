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
"""Manage a session's lease on the native allocator's expandable segments."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import logging
from pathlib import Path
from threading import Lock
import weakref

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_config_reader() -> Callable[[], bool | None] | None:
    """Return the native configuration getter, or None without a supported toolchain."""
    try:
        # Optional extension loading must not add a compiler dependency to import.
        from torch.utils.cpp_extension import load  # pylint: disable=import-outside-toplevel

        module = load(
            name="hyperoffload_allocator_config",
            sources=[str(Path(__file__).with_name("allocator_config.cpp"))],
            extra_cflags=["-O2"], verbose=False,
        )
    except (ImportError, OSError, RuntimeError) as error:
        logger.warning("HyperOffload allocator configuration helper unavailable: %s", error)
        return None
    return module.expandable_segments


class _AllocatorState:
    """Share the process-wide allocator setting across overlapping session owners."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._users = 0
        self._previous = False
        self._setter = None
        self._reader = None

    def enter(
        self, reader: Callable[[], bool | None], setter: Callable[[str], None],
        retire_cache: Callable[[], None],
    ) -> bool:
        """Acquire a lease, enabling and retiring old unused segments only once.

        Args:
            reader: Getter of the initialized backend's actual current setting.
            setter: Native partial allocator configuration update.
            retire_cache: Release only unused cached segments after a policy change.
        """
        with self._lock:
            if not self._users:
                previous = reader()
                if previous is None:
                    return False
                if not previous:
                    try:
                        setter("expandable_segments:True")
                    except (RuntimeError, ValueError, TypeError) as error:
                        # A backend may reject a private setting after applying
                        # it. Only fall back once its original state is restored.
                        if reader() != previous:
                            setter("expandable_segments:False")
                        logger.warning(
                            "HyperOffload allocator policy unavailable; keeping original settings: %s", error,
                        )
                        return False
                    try:
                        # Model construction can leave fixed-size split segments
                        # that would otherwise compete with the new expandable ones.
                        retire_cache()
                    except BaseException:
                        setter("expandable_segments:False")
                        raise
                self._previous = previous
                self._setter = setter
                self._reader = reader
            self._users += 1
            return True

    def exit(self) -> None:
        """Release a lease and restore the original value after the final owner."""
        with self._lock:
            self._users -= 1
            if not self._users:
                try:
                    if not self._previous and self._reader():
                        self._setter("expandable_segments:False")
                finally:
                    self._setter = None
                    self._reader = None


_allocator_state = _AllocatorState()


class SessionAllocator:
    """Keep native allocation policy consistent across forward, backward and optimizer.

    The setting is process-wide, including allocations outside a session's context.
    Its lease follows the session object's lifetime, since optimizer work usually
    runs between context entries. Explicit close or object destruction releases
    the lease. Live tensors remain valid after restoration. Unsupported APIs leave
    the backend unchanged; this policy does not replace the native allocator.
    """

    def __init__(self) -> None:
        """Defer device-policy setup until the first accelerator execution."""
        self._initialized = False
        self._finalizer: weakref.finalize | None = None

    def activate(self) -> None:
        """Acquire once, without initializing an accelerator for CPU-only execution."""
        if self._initialized or not torch.accelerator.is_available():
            return
        device = torch.accelerator.current_accelerator()
        if device is None or device.type not in ("cuda", "npu"):
            return
        backend = torch.get_device_module(device)
        # Device selection/model allocation must precede session execution. A
        # CPU unit test in an NPU-capable process must not initialize hardware.
        if not backend.is_initialized():
            return
        memory = backend.memory
        setter = getattr(memory, "_set_allocator_settings", None)
        reader = _get_config_reader() if setter is not None else None
        if reader is not None and _allocator_state.enter(reader, setter, backend.empty_cache):
            self._finalizer = weakref.finalize(self, _allocator_state.exit)
        self._initialized = True

    def close(self) -> None:
        """Release this lease; a subsequent activation can acquire it again."""
        try:
            if self._finalizer is not None:
                self._finalizer()
        finally:
            self._finalizer = None
            self._initialized = False
