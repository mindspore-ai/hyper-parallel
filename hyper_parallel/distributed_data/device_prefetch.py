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
"""Trainer- and producer-owned Host-to-device batch prefetch."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
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
        if value.device.type != "cpu":
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


def _pin_memory(value: Any) -> Any:
    """Pin final collated CPU tensors without changing metadata or dtype."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and value.layout == torch.strided and not value.is_pinned():
            return value.pin_memory()
        return value
    if isinstance(value, Mapping):
        return {key: _pin_memory(item) for key, item in value.items()}
    if isinstance(value, tuple):
        pinned = tuple(_pin_memory(item) for item in value)
        return type(value)(*pinned) if hasattr(value, "_fields") else pinned
    if isinstance(value, list):
        return [_pin_memory(item) for item in value]
    pin = getattr(value, "pin_memory", None)
    return pin() if callable(pin) else value


class DevicePrefetchedStep:
    """Retain Host statistics inputs while handing ready device microbatches to training."""

    def __init__(
            self,
            cpu_micro_batches: list[Any],
            device_micro_batches: list[Any],
            ready_event: Any,
            device: torch.device,
    ) -> None:
        """Store both views and the producer's completed copy event."""
        self.cpu_micro_batches = cpu_micro_batches
        self.device_micro_batches = device_micro_batches
        self.ready_event = ready_event
        self.device = device

    def take_microbatch(self, index: int) -> Any:
        """Transfer one microbatch's lifetime to the current consumer stream.

        Args:
            index: Position of an unconsumed microbatch in this step.

        Returns:
            Device microbatch, with retained CPU metadata left untouched.
        """
        accelerator = getattr(torch, self.device.type)
        stream = accelerator.current_stream(self.device)
        stream.wait_event(self.ready_event)
        micro_batch = self.device_micro_batches[index]
        _record_stream(micro_batch, stream)
        self.device_micro_batches[index] = None
        return micro_batch


class DeviceStepPrefetcher:
    """Stage a complete step inside the DataLoader's existing producer thread.

    This is not an additional iterator, queue, or thread. Packing and payload
    exchange finish before this stage runs. The producer waits only for its
    own H2D event before publishing a step, while training uses the original
    Host view for metrics and takes device microbatches immediately before use.
    The trainer-owned :class:`DeviceBatchPrefetcher` API remains independent.
    """

    def __init__(
            self,
            device: Any,
            *,
            move_fn: Callable[[Any, torch.device], Any] | None = None,
            pin_memory: bool = False,
            pin_fn: Callable[[Any], Any] | None = None,
            profile_context_fn: Callable[[], AbstractContextManager] | None = None,
    ) -> None:
        """Configure final input transfer without allocating a device stream.

        Args:
            device: Target CUDA or NPU device, independent of payload transport.
            move_fn: Optional per-microbatch field mapping/H2D callback. The
                default recursively moves tensors and standard containers.
            pin_memory: Re-pin final packed inputs before H2D when enabled.
            pin_fn: Optional per-microbatch pin policy, matching ``move_fn``.
            profile_context_fn: Optional application profiling context factory.
        """
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "npu"):
            raise ValueError("DeviceStepPrefetcher requires a CUDA or NPU device.")
        self._accelerator = _accelerator_module(self.device)
        self.pin_memory = pin_memory
        self._move_fn = _move_to_device if move_fn is None else move_fn
        self._pin_fn = _pin_memory if pin_fn is None else pin_fn
        self._profile_context_fn = nullcontext if profile_context_fn is None else profile_context_fn
        self._copy_stream = None

    def __call__(self, cpu_micro_batches: list[Any]) -> DevicePrefetchedStep:
        """Stage final microbatches and publish only after their H2D completes.

        Args:
            cpu_micro_batches: Final collated step, retained for Host metering.

        Returns:
            Host/device views of the step with a completed copy event.
        """
        accelerator = self._accelerator
        # The local loader may create a fresh producer thread each step; device
        # selection is thread-local even when payload exchange uses only Gloo.
        accelerator.set_device(self.device)
        if self._copy_stream is None:
            self._copy_stream = accelerator.Stream(device=self.device)
        with self._profile_context_fn():
            staging = (
                [self._pin_fn(micro_batch) for micro_batch in cpu_micro_batches]
                if self.pin_memory else cpu_micro_batches
            )
            device_micro_batches = []
            try:
                with accelerator.stream(self._copy_stream):
                    for micro_batch in staging:
                        device_micro_batches.append(self._move_fn(micro_batch, self.device))
                    ready_event = accelerator.Event()
                    ready_event.record(self._copy_stream)
                ready_event.synchronize()
            except BaseException:
                # Earlier copies may still read pinned staging when a later
                # launch fails. Drain only this stream before releasing sources.
                self._copy_stream.synchronize()
                raise
        return DevicePrefetchedStep(cpu_micro_batches, device_micro_batches, ready_event, self.device)


__all__ = ["DeviceBatchPrefetcher", "DevicePrefetchedStep", "DeviceStepPrefetcher"]
