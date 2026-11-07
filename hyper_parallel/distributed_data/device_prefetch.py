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
"""Pinned Host staging and asynchronous device copies shared by data loaders."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import


def _get_device_module(device: torch.device) -> Any:
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
    """Hand the complete device batch and its copy dependency to training."""

    def __init__(
            self,
            device_batch: Any,
            ready_event: Any,
            device: torch.device,
    ) -> None:
        """Store device inputs while the prefetcher owns pending Host sources."""
        self.device_batch = device_batch
        self.ready_event = ready_event
        self.device = device

    def take_batch(self) -> Any:
        """Transfer the complete batch's lifetime to the current consumer stream.

        Returns:
            Device batch, with non-tensor metadata left untouched.
        """
        device_module = getattr(torch, self.device.type)
        stream = device_module.current_stream(self.device)
        stream.wait_event(self.ready_event)
        batch = self.device_batch
        _record_stream(batch, stream)
        self.device_batch = None
        return batch


class DeviceStepPrefetcher:
    """Separate Host pinning from caller-owned asynchronous device copies."""

    def __init__(self, device: Any) -> None:
        """Configure final input transfer without allocating a device stream.

        Args:
            device: Target CUDA or NPU device, independent of payload transport.
        """
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "npu"):
            raise ValueError("DeviceStepPrefetcher requires a CUDA or NPU device.")
        self._device_module = _get_device_module(self.device)
        self._copy_stream = None
        self._pending_staging: list[tuple[Any, Any]] = []

    def close(self) -> None:
        """Drain pending copies before dropping their pinned source storage."""
        for event, _ in self._pending_staging:
            event.synchronize()
        self._pending_staging.clear()

    def __call__(self, cpu_batch: Any) -> DevicePrefetchedStep:
        """Enqueue the final batch and publish its copy-completion event.

        Args:
            cpu_batch: Final collated batch to pin and copy.

        Returns:
            Device batch with a copy-completion event. The
            caller does not wait for that event; the consumer stream
            orders itself with ``wait_event`` in ``take_batch``.
        """
        return self.stage_prepared(self.prepare_host(cpu_batch))

    @staticmethod
    def prepare_host(cpu_batch: Any) -> Any:
        """Pin a constructed batch on its Host producer before device staging.

        Args:
            cpu_batch: Collated batch whose CPU tensor leaves need pinned storage.

        Returns:
            The batch structure with pinned CPU tensors.
        """
        return _pin_memory(cpu_batch)

    def stage_prepared(self, staging: Any) -> DevicePrefetchedStep:
        """Enqueue an already-pinned batch without CPU allocation or copying for pinning.

        Args:
            staging: Pinned batch prepared by the Host producer.

        Returns:
            Device input and its copy event. This prefetcher retains pending Host sources.
        """
        device_module = self._device_module
        # The local loader may create a fresh producer thread each step; device
        # selection is thread-local even when payload exchange uses only Gloo.
        device_module.set_device(self.device)
        if self._copy_stream is None:
            self._copy_stream = device_module.Stream(device=self.device)
        self._pending_staging = [(event, batch) for event, batch in self._pending_staging if not event.query()]
        try:
            with device_module.stream(self._copy_stream):
                device_batch = _move_to_device(staging, self.device)
                ready_event = device_module.Event()
                ready_event.record(self._copy_stream)
        except BaseException:
            # Earlier copies may still read pinned staging when a later
            # launch fails. Drain only this stream before releasing sources.
            self._copy_stream.synchronize()
            raise
        self._pending_staging.append((ready_event, staging))
        return DevicePrefetchedStep(device_batch, ready_event, self.device)


def _resolve_device(device: Any = None, *, communication_backend: str = "hccl") -> torch.device:
    """Keep Gloo on CPU by default; otherwise select the current accelerator."""
    if device is None and communication_backend == "gloo":
        return torch.device("cpu")
    if device is None:
        for device_type in ("npu", "cuda"):
            device_module = getattr(torch, device_type, None)
            if device_module is not None and device_module.is_available():
                device = torch.device(device_type, device_module.current_device())
                break
        else:
            device = "cpu"
    return torch.device(device)


def _create_device_prefetcher(device: Any = None) -> DeviceStepPrefetcher | None:
    """Create an H2D copy stage only when an accelerator is selected."""
    device = _resolve_device(device)
    if device.type == "cpu":
        return None
    return DeviceStepPrefetcher(device)


__all__ = ["DeviceStepPrefetcher"]
