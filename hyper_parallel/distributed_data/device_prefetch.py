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
"""Producer-owned Host-to-device prefetch shared by all data loaders."""
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
            staging_micro_batches: list[Any] | None = None,
    ) -> None:
        """Retain batch views and pinned sources while device copies are pending."""
        self.cpu_micro_batches = cpu_micro_batches
        self.device_micro_batches = device_micro_batches
        self.ready_event = ready_event
        self.device = device
        self._staging_micro_batches = staging_micro_batches

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
    exchange finish before this stage runs. The consumer stream waits for the
    H2D event before using inputs, while training uses the original
    Host view for metrics and takes device microbatches immediately before use.
    """

    def __init__(
            self,
            device: Any,
            *,
            move_fn: Callable[[Any, torch.device], Any] | None = None,
    ) -> None:
        """Configure final input transfer without allocating a device stream.

        Args:
            device: Target CUDA or NPU device, independent of payload transport.
            move_fn: Optional per-microbatch field mapping/H2D callback. The
                default recursively moves tensors and standard containers.
        """
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "npu"):
            raise ValueError("DeviceStepPrefetcher requires a CUDA or NPU device.")
        if move_fn is not None and not callable(move_fn):
            raise ValueError("move_fn must be callable or None.")
        self._accelerator = _accelerator_module(self.device)
        self._move_fn = _move_to_device if move_fn is None else move_fn
        self._copy_stream = None
        self._pending_staging: list[tuple[Any, list[Any]]] = []

    def close(self) -> None:
        """Drain pending copies before dropping their pinned source storage."""
        for event, _ in self._pending_staging:
            event.synchronize()
        self._pending_staging.clear()

    def __call__(self, cpu_micro_batches: list[Any]) -> DevicePrefetchedStep:
        """Enqueue final microbatches and publish their copy-completion event.

        Args:
            cpu_micro_batches: Final collated step, retained for Host metering.

        Returns:
            Host/device views of the step with a copy-completion event.  The
            producer thread does not wait for that event; the consumer stream
            orders itself with ``wait_event`` in ``take_microbatch``.
        """
        accelerator = self._accelerator
        # The local loader may create a fresh producer thread each step; device
        # selection is thread-local even when payload exchange uses only Gloo.
        accelerator.set_device(self.device)
        if self._copy_stream is None:
            self._copy_stream = accelerator.Stream(device=self.device)
        self._pending_staging = [(event, batch) for event, batch in self._pending_staging if not event.query()]
        staging = [_pin_memory(micro_batch) for micro_batch in cpu_micro_batches]
        device_micro_batches = []
        try:
            with accelerator.stream(self._copy_stream):
                for micro_batch in staging:
                    device_micro_batches.append(self._move_fn(micro_batch, self.device))
                ready_event = accelerator.Event()
                ready_event.record(self._copy_stream)
        except BaseException:
            # Earlier copies may still read pinned staging when a later
            # launch fails. Drain only this stream before releasing sources.
            self._copy_stream.synchronize()
            raise
        self._pending_staging.append((ready_event, staging))
        return DevicePrefetchedStep(
            cpu_micro_batches,
            device_micro_batches,
            ready_event,
            self.device,
            staging_micro_batches=staging,
        )


def _resolve_device(device: Any = None, *, communication_backend: str = "hccl") -> torch.device:
    """Keep Gloo on CPU by default; otherwise select the current accelerator."""
    if device is None and communication_backend == "gloo":
        return torch.device("cpu")
    if device is None:
        for device_type in ("npu", "cuda"):
            accelerator = getattr(torch, device_type, None)
            if accelerator is not None and accelerator.is_available():
                device = torch.device(device_type, accelerator.current_device())
                break
        else:
            device = "cpu"
    return torch.device(device)


def _create_device_prefetcher(
        device: Any = None,
        move_fn: Callable[[Any, torch.device], Any] | None = None,
) -> DeviceStepPrefetcher | None:
    """Create producer-owned H2D only when an accelerator is selected."""
    device = _resolve_device(device)
    if device.type == "cpu":
        return None
    return DeviceStepPrefetcher(device, move_fn=move_fn)


__all__ = ["DeviceStepPrefetcher"]
