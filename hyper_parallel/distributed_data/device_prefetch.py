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
"""Trainer-side asynchronous Host-to-device batch prefetch."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import


def _accelerator_module(device: torch.device) -> Any:
    """Return the PyTorch device module that owns streams and events."""
    module = getattr(torch, device.type, None)
    required = ("Event", "Stream", "current_stream", "stream")
    if module is None or any(not callable(getattr(module, name, None)) for name in required):
        raise ValueError(
            f"Device {device} does not expose PyTorch accelerator Stream/Event APIs."
        )
    return module


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors, batch objects, and standard containers recursively."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    to_device = getattr(value, "to", None)
    if callable(to_device):
        return to_device(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        moved = tuple(_move_to_device(item, device) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return moved
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _record_stream(value: Any, stream: Any, visited: set[int] | None = None) -> None:
    """Associate device batch storage with its consumer stream."""
    if visited is None:
        visited = set()
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)

    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
        return
    record_stream = getattr(value, "record_stream", None)
    if callable(record_stream):
        record_stream(stream)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _record_stream(item, stream, visited)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream, visited)
        return

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        for item in attributes.values():
            _record_stream(item, stream, visited)


class DeviceBatchPrefetcher:
    """Maintain one Trainer-owned batch on an accelerator copy stream.

    The distributed DataLoader prepares a complete Host batch. The Trainer
    calls :meth:`prefetch` after launching the current backward pass, then
    calls :meth:`wait` before the next model or broadcast operation consumes
    that batch. An optional ``prepare_fn`` performs CPU-only model-specific
    work, such as extracting a CP UND attention-mask block, before H2D.
    """

    def __init__(
            self,
            device: Any,
            *,
            prepare_fn: Callable[[Any], Any] | None = None,
            move_fn: Callable[[Any, torch.device], Any] | None = None,
    ) -> None:
        """Initialize an empty one-batch device slot.

        Args:
            device: Target CUDA or NPU device.
            prepare_fn: Optional CPU transformation run before H2D.
            move_fn: Optional H2D callback. It must enqueue non-blocking copies
                on the current copy stream. The default supports tensors,
                objects with ``to(device, non_blocking=True)``, and standard
                nested containers.

        Raises:
            ValueError: If the device or callbacks are invalid.
        """
        try:
            normalized_device = torch.device(device)
        except Exception as exc:
            raise ValueError(f"device is invalid: {device!r}.") from exc
        if normalized_device.type == "cpu":
            raise ValueError("DeviceBatchPrefetcher requires an accelerator device, not CPU.")
        if prepare_fn is not None and not callable(prepare_fn):
            raise ValueError("prepare_fn must be callable or None.")
        if move_fn is not None and not callable(move_fn):
            raise ValueError("move_fn must be callable or None.")
        self._device = normalized_device
        self._prepare_fn = prepare_fn
        self._move_fn = _move_to_device if move_fn is None else move_fn
        self._accelerator = _accelerator_module(self._device)
        self._copy_stream = None
        self._pending_batch = None
        self._ready_event = None
        self._has_pending = False

    @property
    def has_pending(self) -> bool:
        """Return whether one prefetched device batch awaits consumption."""
        return self._has_pending

    def prefetch(self, host_batch: Any) -> None:
        """Prepare and asynchronously move one Host batch to the device.

        Args:
            host_batch: Complete Host batch returned by the distributed loader.

        Raises:
            ValueError: If the one-batch slot is already occupied.
        """
        if self._has_pending:
            raise ValueError("Cannot prefetch a second device batch before consuming the pending batch.")
        prepared_batch = self._prepare_fn(host_batch) if self._prepare_fn is not None else host_batch
        if self._copy_stream is None:
            self._copy_stream = self._accelerator.Stream(device=self._device)
        with self._accelerator.stream(self._copy_stream):
            device_batch = self._move_fn(prepared_batch, self._device)
            ready_event = self._accelerator.Event()
            ready_event.record(self._copy_stream)
        self._pending_batch = device_batch
        self._ready_event = ready_event
        self._has_pending = True

    def wait(self) -> Any:
        """Order the consumer stream after H2D and return the ready batch.

        Returns:
            The prefetched device batch.

        Raises:
            ValueError: If no device batch is pending.
        """
        if not self._has_pending or self._ready_event is None:
            raise ValueError("No prefetched device batch is pending.")
        current_stream = self._accelerator.current_stream(self._device)
        self._ready_event.wait(current_stream)
        batch = self._pending_batch
        _record_stream(batch, current_stream)
        self._pending_batch = None
        self._ready_event = None
        self._has_pending = False
        return batch


__all__ = ["DeviceBatchPrefetcher"]
