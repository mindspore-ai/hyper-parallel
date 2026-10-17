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
"""Model-independent logical storage tracking with native snapshot output."""

import copy
import functools
import os
import pickle
import sys
import tempfile
import weakref
from pathlib import Path
from typing import Any, Iterable

import torch
from torch._guards import active_fake_mode
# Register shape-only collective implementations for FakeTensor execution.
import torch.distributed._tools.fake_collectives  # pylint: disable=unused-import
from torch.distributed._tools.common_utils import get_untyped_storages
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


_ADDRESS_BASE = 0xF00000000000
_ADDRESS_ALIGNMENT = 512


class LogicalMemoryTracker(TorchDispatchMode):
    """Observe storage capacity and lifetime without classifying tensor roles.

    Args:
        device_type: Logical device type of tensors being simulated.
        device_index: Device index used for CPU FakeTensor fallback.

    Note:
        Segments are synthetic, one per live allocation. There is no caching,
        fragmentation, workspace, or asynchronous stream simulation.
    """

    def __init__(self, device_type: str, device_index: int = 0) -> None:
        """Initialize empty storage identities, allocation history, and totals."""
        super().__init__()
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        self.device_type = device_type
        self.device_index = device_index
        self._allocations: dict[int, dict[str, Any]] = {}
        self._storage_refs: dict[int, Any] = {}
        self._traces: list[list[dict[str, Any]]] = []
        self._current: dict[int, int] = {}
        self._peaks: dict[int, int] = {}
        self._peak_bytes = 0
        self._next_address = _ADDRESS_BASE
        self._previous_resize = None
        self._recording = False
        self._stack_cache: dict[tuple, list[dict[str, Any]]] = {}

    def __enter__(self) -> "LogicalMemoryTracker":
        """Start dispatch tracking and observe direct storage resizes."""
        if self._recording:
            raise RuntimeError("LogicalMemoryTracker is already recording")
        super().__enter__()
        self._recording = True
        self._previous_resize = torch.UntypedStorage.resize_
        previous_resize = self._previous_resize

        @functools.wraps(previous_resize)
        def resize(storage: Any, size: int) -> Any:
            """Record capacity transitions after the storage resize completes."""
            result = previous_resize(storage, size)
            key = int(storage._cdata)  # pylint: disable=protected-access
            allocation = self._allocations.get(key)
            if allocation is not None:
                self._observe_storage(storage, allocation["device"])
            return result

        torch.UntypedStorage.resize_ = resize
        return self

    def __exit__(self, *args: Any) -> Any:
        """Restore the storage resize hook and freeze the recorded state."""
        torch.UntypedStorage.resize_ = self._previous_resize
        self._previous_resize = None
        self._recording = False
        return super().__exit__(*args)

    def _frames(self) -> list[dict[str, Any]]:
        frame = sys._getframe().f_back  # pylint: disable=protected-access
        frames = []
        while frame is not None and len(frames) < 64:
            code = frame.f_code
            if code.co_filename != __file__:
                frames.append((code.co_filename, frame.f_lineno, code.co_name))
            frame = frame.f_back
        key = tuple(frames)
        if key not in self._stack_cache:
            self._stack_cache[key] = [
                {"filename": filename, "line": line, "name": name}
                for filename, line, name in frames
            ]
        return self._stack_cache[key]

    def _event(self, action: str, allocation: dict[str, Any]) -> None:
        device = allocation["device"]
        while len(self._traces) <= device:
            self._traces.append([])
        self._traces[device].append({
            "action": action,
            "addr": allocation["address"],
            "size": allocation["size"],
            "stream": 0,
            "frames": allocation["frames"],
            "pool_id": (0, 0),
        })

    def _release(self, key: int) -> None:
        allocation = self._allocations.pop(key, None)
        self._storage_refs.pop(key, None)
        if allocation is None:
            return
        if allocation["size"] == 0:
            return
        allocation = {**allocation, "frames": self._frames()}
        self._event("free_requested", allocation)
        self._event("free_completed", allocation)
        self._event("segment_free", allocation)
        self._current[allocation["device"]] -= allocation["size"]

    def _observe_storage(self, storage: Any, device: int) -> None:
        key = int(storage._cdata)  # pylint: disable=protected-access
        size = int(storage.nbytes())
        allocation = self._allocations.get(key)
        if allocation is not None:
            if allocation["size"] == size:
                return
            self._release(key)
        tracker_ref = weakref.ref(self)

        def released(storage_ref: Any) -> None:
            """Close the allocation only if this is its current storage reference."""
            tracker = tracker_ref()
            if tracker is not None and tracker._recording and tracker._storage_refs.get(key) is storage_ref:
                tracker._release(key)

        # Keep zero-capacity storage identities so a later resize can be observed.
        self._storage_refs[key] = weakref.ref(storage, released)
        if size == 0:
            self._allocations[key] = {
                "device": device, "size": 0, "address": 0, "frames": [],
            }
            return
        address = self._next_address
        self._next_address += ((size + _ADDRESS_ALIGNMENT - 1) // _ADDRESS_ALIGNMENT) * _ADDRESS_ALIGNMENT
        allocation = {"device": device, "size": size, "address": address, "frames": self._frames()}
        self._allocations[key] = allocation
        self._event("segment_alloc", allocation)
        self._event("alloc", allocation)
        self._current[device] = self._current.get(device, 0) + size
        self._peaks[device] = max(self._peaks.get(device, 0), self._current[device])
        self._peak_bytes = max(self._peak_bytes, sum(self._current.values()))

    def track(self, tensors: Iterable[torch.Tensor]) -> None:
        """Register existing tensors without retaining them or their storages.

        Args:
            tensors: Existing inputs, state, or newly returned tensors.
        """
        if not self._recording:
            raise RuntimeError("Enter LogicalMemoryTracker before registering tensors")
        for tensor in tensors:
            if tensor.device.type != self.device_type:
                continue
            device = tensor.device.index
            if device is None:
                device = self.device_index
            for storage in get_untyped_storages(tensor):
                self._observe_storage(storage, device)

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple = (), kwargs: Any = None) -> Any:
        del types
        input_values, _ = tree_flatten((args, kwargs))
        # FakeTensor constant propagation must not execute collective kernels
        # on tiny materialized scalars in the virtual process group.
        if str(func).startswith("c10d."):
            for value in input_values:
                if isinstance(value, torch.Tensor) and hasattr(value, "fake_mode"):
                    value.constant = None
        self.track(value for value in input_values if isinstance(value, torch.Tensor))
        # Fake wait_tensor's Meta kernel returns a new storage, unlike a real wait.
        if func is torch.ops._c10d_functional.wait_tensor.default and active_fake_mode() is not None:
            result = args[0]
        else:
            result = func(*args, **(kwargs or {}))
        output_values, _ = tree_flatten(result)
        self.track(value for value in (*input_values, *output_values) if isinstance(value, torch.Tensor))
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return an independent native-compatible snapshot of logical memory.

        Returns:
            Native segments and device traces, with logical summary and
            simulation assumptions. Event list order is the logical timeline.
        """
        segments = []
        for allocation in self._allocations.values():
            size = allocation["size"]
            if size == 0:
                continue
            segments.append({
                "device": allocation["device"],
                "address": allocation["address"],
                "total_size": size,
                "allocated_size": size,
                "active_size": size,
                "requested_size": size,
                "stream": 0,
                "segment_type": "small" if size <= 2 ** 20 else "large",
                "segment_pool_id": (0, 0),
                "is_expandable": False,
                "frames": allocation["frames"],
                "blocks": [{
                    "address": allocation["address"],
                    "size": size,
                    "requested_size": size,
                    "state": "active_allocated",
                    "frames": allocation["frames"],
                }],
            })
        # Copies isolate saved snapshots from subsequent events and releases.
        return copy.deepcopy({
            "segments": sorted(segments, key=lambda segment: (segment["device"], segment["address"])),
            "device_traces": self._traces,
            "summary": {
                "current_bytes": sum(self._current.values()),
                "peak_bytes": self._peak_bytes,
                "peak_gib": self._peak_bytes / 2 ** 30,
                "devices": {
                    device: {"current_bytes": current_bytes, "peak_bytes": self._peaks[device]}
                    for device, current_bytes in self._current.items()
                },
            },
            "simulation": {
                "memory": "logical storage bytes",
                "segments": "synthetic, one per allocation, immediately returned on free",
                "timeline": "event order, not wall-clock time",
                "streams": "synchronous, stream 0",
                "workspace": "not simulated",
                "fragmentation": "not simulated",
            },
        })


def write_memory_snapshot(snapshot: dict[str, Any], output_path: str) -> str:
    """Atomically write a native-compatible pickle snapshot.

    Args:
        snapshot: LogicalMemoryTracker snapshot, optionally with run metadata.
        output_path: Destination pickle path.

    Returns:
        Absolute destination path.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "wb") as snapshot_file:
            # The browser viewer supports a subset of pickle, with protocol 4
            # also accepted by older deployed MemoryViz versions.
            pickle.dump(snapshot, snapshot_file, protocol=4)
        os.replace(temporary_path, destination)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return str(destination)


__all__ = ["LogicalMemoryTracker", "write_memory_snapshot"]
