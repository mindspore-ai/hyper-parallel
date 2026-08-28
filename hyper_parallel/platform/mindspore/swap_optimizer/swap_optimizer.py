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
"""MindSpore optimizer state swap wrapper."""
# pylint: disable=protected-access

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import mindspore as ms
from mindspore.graph.api import _no_grad

from hyper_parallel.core.optimizer.swap_optimizer_base import (
    STATE_KEYS,
    PipelineSwapRuntime,
    SwapSlot,
    _iter_unique_slots,
)
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.mindspore.swap_optimizer.adapters import (
    MindFormersAdamWAdapter,
    MindSporeNativeAdamAdapter,
    MindSporeNativeAdamWAdapter,
)

platform = get_platform()
_PACKED_ALIGNMENT_BYTES = 512


@dataclass
class _PackedBatchRegion:
    """One persistent dtype buffer transferred for a pipeline batch."""

    dtype: Any
    numel: int
    slots: List[SwapSlot]


@dataclass
class _PackedBatchPlan:
    """Persistent packed transfer layout for one optimizer pipeline batch."""

    regions: Dict[Any, _PackedBatchRegion] = field(default_factory=dict)


@dataclass
class _StagingArena:
    """One step-local raw NPU allocation and its dtype-specific views."""

    raw_buffer: Any
    dtype_views: Dict[Any, Any] = field(default_factory=dict)


class MindSporeSwapRuntime(PipelineSwapRuntime):
    """MindSpore tensor storage/copy runtime."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._packed_enabled = bool(getattr(config, "packed_swap", True))
        self._host_buffers: List[Dict[Any, Any]] = []
        self._host_layout_signature = ()
        self._packed_batch_plans: List[_PackedBatchPlan] = []
        self._staging_arenas: List[Optional[_StagingArena]] = [None, None]
        self._packed_ready_events: Dict[int, Any] = {}
        self._packed_offload_events: Dict[int, Any] = {}
        self._packed_tail_event: Optional[Any] = None

    @property
    def packed_enabled(self) -> bool:
        """Return whether packed MindSpore optimizer swap is enabled."""
        return self._packed_enabled

    def populate_slot_metadata(self, slot: SwapSlot, template: Any) -> None:
        """Populate stable logical metadata before the source storage is released."""
        storage_tensor = self._storage_tensor(template)
        slot.shape = tuple(storage_tensor.shape)
        slot.dtype = storage_tensor.dtype
        slot.device = storage_tensor.device
        slot.numel = int(storage_tensor.numel())
        slot.storage_nbytes = slot.numel * int(storage_tensor.itemsize)

    def prepare_packed_host(self, batches: Sequence[Sequence[Any]]) -> bool:
        """Pack optimizer states into persistent pinned CPU buffers by batch and dtype.

        Returns:
            Whether the persistent host layout was rebuilt.
        """
        if not self._packed_enabled:
            return False
        batch_lists = [list(batch) for batch in batches]
        signature = self._packed_layout_signature(batch_lists)
        if signature == self._host_layout_signature:
            return False

        new_buffers: List[Dict[Any, Any]] = []
        new_plans: List[_PackedBatchPlan] = []
        slot_bindings: List[tuple[SwapSlot, int, Any]] = []
        layout_slot_ids = set()
        for batch in batch_lists:
            slots_by_dtype: Dict[Any, List[SwapSlot]] = {}
            for unit in batch:
                for slot in unit.slots:
                    if not slot.swappable or not slot.packed:
                        continue
                    slot_id = id(slot)
                    if slot_id in layout_slot_ids:
                        raise ValueError(
                            f"Packed swap slot {slot.name!r} appears in more than one host batch."
                        )
                    if slot.state != "host":
                        raise RuntimeError(
                            f"Packed swap slot {slot.name!r} must be idle on host before layout preparation."
                        )
                    # The D2H copy may still be pending after the compute stream
                    # ordered storage release.  This is the host-consumption
                    # boundary for the mirror, so wait here before reading it.
                    if slot.event is not None:
                        self.wait_event(slot.event, None)
                        slot.event = None
                    if slot.cpu_tensor is None:
                        raise RuntimeError(
                            f"Packed swap slot {slot.name!r} has no CPU mirror before host layout preparation."
                        )
                    layout_slot_ids.add(slot_id)
                    slots_by_dtype.setdefault(slot.dtype, []).append(slot)

            batch_buffers: Dict[Any, Any] = {}
            batch_regions: Dict[Any, _PackedBatchRegion] = {}
            for dtype, dtype_slots in slots_by_dtype.items():
                total_numel = sum(slot.numel for slot in dtype_slots)
                host_buffer = ms.mint.empty(
                    (total_numel,), dtype=dtype, device="cpu", pin_memory=True
                )
                batch_buffers[dtype] = host_buffer
                host_offset = 0
                for slot in dtype_slots:
                    host_view = self._make_cpu_storage_view(host_buffer, host_offset, slot.shape)
                    self._copy_cpu_tensor(host_view, slot.cpu_tensor)
                    slot_bindings.append((slot, host_offset, host_view))
                    host_offset += slot.numel
                batch_regions[dtype] = _PackedBatchRegion(dtype, total_numel, dtype_slots)
            new_buffers.append(batch_buffers)
            new_plans.append(_PackedBatchPlan(batch_regions))

        for slot, host_offset, host_view in slot_bindings:
            slot.host_offset = host_offset
            slot.cpu_tensor = host_view
            slot.tensor = host_view
            slot.state = "host"
            slot.event = None
        self._host_buffers = new_buffers
        self._packed_batch_plans = new_plans
        self._host_layout_signature = signature
        return True

    def is_swappable_tensor(self, tensor: Any, min_numel: int) -> bool:
        """Return whether ``tensor`` can participate in swap."""
        storage_tensor = self._storage_tensor(tensor)
        if not isinstance(storage_tensor, ms.Tensor):
            return False
        if not _is_float_dtype(storage_tensor.dtype):
            return False
        if int(storage_tensor.numel()) < int(min_numel):
            return False
        if not storage_tensor.is_contiguous():
            return False
        if _is_cpu_tensor(storage_tensor):
            return False
        try:
            storage = storage_tensor.untyped_storage()
            expected = int(storage_tensor.numel()) * int(storage_tensor.itemsize)
            if storage.size() != expected:
                return False
        except (AttributeError, RuntimeError):
            return False
        return True

    def is_packable_tensor(self, tensor: Any, min_numel: int) -> bool:
        """Return whether a CPU- or NPU-resident state can enter packed storage."""
        storage_tensor = self._storage_tensor(tensor)
        if not isinstance(storage_tensor, ms.Tensor):
            return False
        if not _is_float_dtype(storage_tensor.dtype):
            return False
        if int(storage_tensor.numel()) < int(min_numel) or not storage_tensor.is_contiguous():
            return False
        try:
            expected = int(storage_tensor.numel()) * int(storage_tensor.itemsize)
            return storage_tensor.untyped_storage().size() == expected
        except (AttributeError, RuntimeError):
            return False

    def storage_nbytes(self, tensor: Any) -> int:
        """Return storage bytes for a MindSpore tensor."""
        if not isinstance(tensor, ms.Tensor):
            return 0
        try:
            return int(tensor.untyped_storage().size())
        except (AttributeError, RuntimeError):
            return int(tensor.numel()) * int(tensor.itemsize)

    def refresh_swappable_slots(self, batch: Any) -> None:
        """Mark state slots that become device-resident after an update."""
        for slot in _iter_unique_slots(batch):
            if slot.swappable or slot.name not in STATE_KEYS:
                continue
            if not self.is_swappable_tensor(slot.tensor, self.config.min_numel):
                continue
            slot.swappable = True
            slot.storage_nbytes = self.storage_nbytes(slot.tensor)
            slot.state = "device"

    def make_cpu_tensor(self, tensor: Any) -> Any:
        """Create a CPU mirror tensor and copy live storage into it."""
        source = self._storage_tensor(tensor)
        cpu_tensor = ms.mint.empty(
            tuple(source.shape), dtype=source.dtype, device="cpu", pin_memory=True
        )
        if _is_cpu_tensor(source):
            # In an Ascend context MindSpore's Tensor.copy()/clone() may place
            # the result on Ascend even when the source tensor is on CPU.
            self._copy_cpu_tensor(cpu_tensor, source)
        else:
            self._copy_storage(cpu_tensor, source)
        return cpu_tensor

    def copy_cpu_tensor(self, target: Any, source: Any) -> None:
        """Copy matching CPU tensors without dispatching a device operator."""
        self._copy_cpu_tensor(target, source)

    def copy_to_device(self, slot: SwapSlot) -> None:
        """Copy CPU mirror into live tensor."""
        if slot.state != "host":
            return
        if slot.cpu_tensor is None:
            return
        self.load_into_tensor(slot.tensor, slot.cpu_tensor)
        slot.state = "device"

    def wait_prefetch_slot(self, slot: SwapSlot) -> None:
        """MindSpore fallback copies are stream ordered."""
        slot.state = "device"

    def copy_to_cpu(self, slot: SwapSlot) -> None:
        """Copy live tensor into CPU mirror."""
        if slot.cpu_tensor is None:
            slot.cpu_tensor = self.make_cpu_tensor(slot.tensor)
        else:
            self._copy_storage(slot.cpu_tensor, slot.tensor)
        slot.state = "d2h"

    def wait_offload_slot(self, slot: SwapSlot) -> None:
        """Release storage after offload."""
        self.release_device_storage(slot)
        slot.state = "host"

    def load_into_tensor(self, tensor: Any, value: Any) -> None:
        """Copy CPU mirror storage into the live tensor storage."""
        self._copy_storage(tensor, value)

    def restore_device_storage(self, slot: SwapSlot) -> None:
        """Restore live tensor storage."""
        storage_tensor = self._storage_tensor(slot.tensor)
        if _is_cpu_tensor(storage_tensor):
            return
        try:
            storage = storage_tensor.untyped_storage()
            if storage.size() != slot.storage_nbytes:
                storage.resize_(slot.storage_nbytes)
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to restore device storage for swap slot {slot.name!r}: {exc}"
            ) from exc

    def release_device_storage(self, slot: SwapSlot) -> None:
        """Release live tensor storage."""
        storage_tensor = self._storage_tensor(slot.tensor)
        if _is_cpu_tensor(storage_tensor):
            return
        try:
            storage = storage_tensor.untyped_storage()
            if storage.size() != 0:
                storage.resize_(0)
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to release device storage for swap slot {slot.name!r}: {exc}"
            ) from exc

    def supports_packed_pipeline(self, batches: Sequence[Sequence[Any]]) -> bool:
        """Return whether every swappable batch slot uses packed host storage."""
        if not self._packed_enabled:
            return False
        if not batches:
            return False

        slots = [slot for batch in batches for unit in batch for slot in unit.slots if slot.swappable]
        if not slots:
            return False

        unpacked_slots = [slot for slot in slots if not slot.packed]
        missing_cpu_slots = [slot for slot in slots if slot.cpu_tensor is None]
        if unpacked_slots or missing_cpu_slots:
            return False

        if self._packed_layout_signature(batches) != self._host_layout_signature:
            raise RuntimeError(
                "Packed MindSpore optimizer batches do not match the persistent host layout."
            )

        return True

    def begin_packed_step(self, batches: Sequence[Sequence[Any]]) -> None:
        """Validate the persistent layout and materialize two step-local staging buffers."""
        ms.runtime.synchronize()
        self._packed_tail_event = None
        if self._packed_layout_signature(batches) != self._host_layout_signature:
            raise RuntimeError(
                "Packed MindSpore optimizer batches changed after host layout preparation."
            )
        self._staging_arenas = [None, None]
        max_numel_by_dtype: Dict[Any, int] = {}
        element_size_by_dtype: Dict[Any, int] = {}
        device = None
        for batch_index, batch_plan in enumerate(self._packed_batch_plans):
            for dtype, region in batch_plan.regions.items():
                max_numel_by_dtype[dtype] = max(max_numel_by_dtype.get(dtype, 0), region.numel)
                element_size_by_dtype[dtype] = int(self._host_buffers[batch_index][dtype].itemsize)
                if device is None and region.slots:
                    device = region.slots[0].device
        if device is None:
            raise RuntimeError("Packed MindSpore optimizer pipeline has no state metadata.")

        dtype_layouts = {}
        byte_offset = 0
        for dtype in sorted(max_numel_by_dtype, key=str):
            byte_offset = self._align_bytes(byte_offset)
            element_size = element_size_by_dtype[dtype]
            num_bytes = max_numel_by_dtype[dtype] * element_size
            dtype_layouts[dtype] = (byte_offset, num_bytes)
            byte_offset += num_bytes
        total_bytes = self._align_bytes(byte_offset)

        for staging_index in range(2):
            arena = self._materialize_staging_arena(staging_index, total_bytes, device)
            arena.dtype_views = {
                dtype: arena.raw_buffer.narrow(0, offset, num_bytes).view(dtype)
                for dtype, (offset, num_bytes) in dtype_layouts.items()
            }
        self._packed_ready_events = {}
        self._packed_offload_events = {}

    def enqueue_packed_prefetch(self, batch_index: int, staging_index: int) -> None:
        """Enqueue one dtype-packed H2D chain."""
        copy_stream = self._get_copy_stream()
        with self.stream_context(copy_stream):
            self._copy_packed_to_device(batch_index, staging_index)
            ready_event = self.record_event(copy_stream)
        self._packed_ready_events[batch_index] = ready_event
        self._packed_tail_event = ready_event

    def wait_packed_prefetch(self, batch_index: int, staging_index: int) -> None:
        """Order the compute stream after a packed prefetch."""
        del staging_index
        event = self._packed_ready_events.get(batch_index)
        if event is None:
            raise RuntimeError(f"Packed MindSpore batch {batch_index} has no ready event.")
        self.wait_event(event, self.current_stream())

    def activate_packed_batch(self, batch_index: int, staging_index: int) -> None:
        """Bind logical optimizer slots to views of one staging arena."""
        batch_plan = self._packed_batch_plans[batch_index]
        arena = self._require_staging_arena(staging_index)
        for region in batch_plan.regions.values():
            dtype_view = arena.dtype_views[region.dtype]
            for slot in region.slots:
                device_view = dtype_view.narrow(0, slot.host_offset, slot.numel).view(slot.shape)
                slot.tensor = device_view
                slot.state = "device"
                slot.event = None

    def enqueue_packed_offload_prefetch(
            self,
            batch_index: int,
            next_index: Optional[int],
            staging_index: int,
    ) -> None:
        """Serialize D2H and the next same-parity H2D on the copy stream."""
        copy_stream = self._get_copy_stream()
        compute_event = self._record_current_stream_event()
        with self.stream_context(copy_stream):
            self.wait_event(compute_event, copy_stream)
            self._copy_packed_to_host(batch_index, staging_index)
            if next_index is not None:
                self._copy_packed_to_device(next_index, staging_index)
            chain_event = self.record_event(copy_stream)
        self._packed_offload_events[batch_index] = chain_event
        self._packed_tail_event = chain_event
        if next_index is not None:
            self._packed_ready_events[next_index] = chain_event

    def wait_packed_offload(self, batch_index: int) -> None:
        """Order the compute stream after a trailing packed transfer chain."""
        event = self._packed_offload_events.get(batch_index)
        if event is None:
            raise RuntimeError(f"Packed MindSpore batch {batch_index} has no offload event.")
        self.wait_event(event, self.current_stream())

    def finish_packed_offload(self, batch_index: int) -> None:
        """Make persistent pinned views authoritative after D2H."""
        for region in self._packed_batch_plans[batch_index].regions.values():
            for slot in region.slots:
                slot.tensor = slot.cpu_tensor
                slot.state = "host"
                slot.event = None

    def release_packed_step_results(self, results: List[Any]) -> None:
        """Drop PyNative update stubs before releasing their staging inputs."""
        results.clear()

    def end_packed_step(self) -> None:
        """Drain transfers, detach all slot views, and destroy both staging arenas."""
        tail_event = getattr(self, "_packed_tail_event", None)
        if tail_event is not None:
            self.wait_event(tail_event, None)
        active_slots = {
            id(slot): slot
            for plan in self._packed_batch_plans
            for region in plan.regions.values()
            for slot in region.slots
            if slot.state == "device"
        }
        # Normal batches offload through _copy_packed_to_host(); recover only interrupted updates here.
        if active_slots:
            compute_stream = self.current_stream()
            synchronize = getattr(compute_stream, "synchronize", None)
            if synchronize is not None:
                synchronize()
            for slot in active_slots.values():
                bounce_buffer = ms.mint.empty(
                    slot.shape, dtype=slot.dtype, device="cpu", pin_memory=True
                )
                self._copy_tensor(bounce_buffer, slot.tensor, non_blocking=False)
                self._copy_cpu_tensor(slot.cpu_tensor, bounce_buffer)
                slot.tensor = slot.cpu_tensor
                slot.state = "host"
                slot.event = None
        all_slots = {
            id(slot): slot
            for plan in self._packed_batch_plans
            for region in plan.regions.values()
            for slot in region.slots
        }
        # A slot view keeps the arena storage alive even after raw_buffer.resize_(0).
        # Detach every slot first, including batches already marked as host-resident.
        for slot in all_slots.values():
            if slot.cpu_tensor is not None:
                slot.tensor = slot.cpu_tensor
                slot.state = "host"
                slot.event = None
        for arena in self._staging_arenas:
            if arena is None:
                continue
            arena.dtype_views = {}
            storage = arena.raw_buffer.untyped_storage()
            if storage.size() != 0:
                storage.resize_(0)
            arena.raw_buffer = None
        # Do not retain zero-sized raw tensors between optimizer steps. MindSpore
        # views may otherwise keep their previous device allocation in the pool.
        self._staging_arenas = [None, None]
        self._packed_ready_events = {}
        self._packed_offload_events = {}
        self._packed_tail_event = None

    def current_stream(self) -> Any:
        """Return the current compute stream."""
        return platform.get_current_stream()

    def new_stream(self) -> Any:
        """Create the copy stream."""
        return platform.new_stream()

    def stream_context(self, stream: Any) -> Any:
        """Return a stream context for ``stream``."""
        if stream is None:
            return contextlib.nullcontext()
        return platform.get_stream_context()(stream)

    def record_event(self, stream: Any = None) -> Any:
        """Record an event on ``stream``."""
        event = platform.new_event()
        if stream is None:
            event.record()
        else:
            event.record(stream)
        return event

    def wait_event(self, event: Any, stream: Any = None) -> None:
        """Make ``stream`` wait for ``event``."""
        if event is None:
            return
        if stream is None:
            event.synchronize()
            return
        event.wait(stream)

    def _copy_storage(self, target: Any, source: Any) -> None:
        """Copy tensor storage without allocating a new tensor object."""
        if isinstance(target, ms.Parameter):
            target = target.data
        if isinstance(source, ms.Parameter):
            source = source.data
        target.untyped_storage().copy_(source.untyped_storage(), non_blocking=True)

    def _copy_tensor(self, target: Any, source: Any, *, non_blocking: bool = True) -> None:
        """Copy logical tensor views while respecting their offsets and lengths."""
        if isinstance(target, ms.Parameter):
            target = target.data
        if isinstance(source, ms.Parameter):
            source = source.data
        target.copy_(source, non_blocking=non_blocking)

    @staticmethod
    def _packed_layout_signature(batches: Sequence[Sequence[Any]]) -> tuple:
        """Return the batch-sensitive identity of a packed host layout."""
        return tuple(
            tuple(
                (
                    getattr(unit, "adapter_index", None),
                    tuple(
                        (id(slot), slot.name, slot.dtype, slot.shape, slot.numel)
                        for slot in unit.slots
                        if slot.swappable and slot.packed
                    ),
                )
                for unit in batch
            )
            for batch in batches
        )

    def _materialize_staging_arena(self, staging_index: int, total_bytes: int, device: Any) -> _StagingArena:
        raw_buffer = platform.empty((total_bytes,), dtype=ms.uint8, device=device)
        arena = _StagingArena(raw_buffer)
        self._staging_arenas[staging_index] = arena
        return arena

    def _copy_packed_to_device(self, batch_index: int, staging_index: int) -> None:
        plan = self._packed_batch_plans[batch_index]
        arena = self._require_staging_arena(staging_index)
        for dtype, region in plan.regions.items():
            host_buffer = self._host_buffers[batch_index][dtype]
            staging_view = arena.dtype_views[dtype].narrow(0, 0, region.numel)
            self._copy_tensor(staging_view, host_buffer)

    def _copy_packed_to_host(self, batch_index: int, staging_index: int) -> None:
        plan = self._packed_batch_plans[batch_index]
        arena = self._require_staging_arena(staging_index)
        for dtype, region in plan.regions.items():
            host_buffer = self._host_buffers[batch_index][dtype]
            staging_view = arena.dtype_views[dtype].narrow(0, 0, region.numel)
            self._copy_tensor(host_buffer, staging_view)

    def _require_staging_arena(self, staging_index: int) -> _StagingArena:
        arena = self._staging_arenas[staging_index]
        if arena is None:
            raise RuntimeError(f"Packed MindSpore staging arena {staging_index} is not materialized.")
        return arena

    def _make_cpu_storage_view(self, host_buffer: Any, element_offset: int, shape: Sequence[int]) -> Any:
        """Create a pinned CPU view directly from the host buffer storage."""
        shape = tuple(shape)
        stride = []
        running_stride = 1
        for dim in reversed(shape):
            stride.append(running_stride)
            running_stride *= int(dim)
        host_view = ms.mint.empty(
            (0,), dtype=host_buffer.dtype, device=host_buffer.device
        ).set_(host_buffer.untyped_storage(), element_offset, shape, tuple(reversed(stride)))
        if not _is_cpu_tensor(host_view):
            raise RuntimeError(
                f"Packed optimizer host view must remain on CPU, but got device {host_view.device!r}."
            )
        return host_view

    def _copy_cpu_tensor(self, target: Any, source: Any) -> None:
        """Copy CPU tensor data directly, including non-zero storage offsets."""
        if isinstance(target, ms.Parameter):
            target = target.data
        if isinstance(source, ms.Parameter):
            source = source.data
        if not _is_cpu_tensor(target) or not _is_cpu_tensor(source):
            raise RuntimeError(
                "Packed host memcpy requires CPU source and target tensors, but got "
                f"target device {getattr(target, 'device', None)!r} and "
                f"source device {getattr(source, 'device', None)!r}."
            )
        if int(target.numel()) != int(source.numel()) or target.dtype != source.dtype:
            raise RuntimeError(
                "Packed host memcpy requires matching tensor sizes and dtypes, but got "
                f"target=({target.numel()}, {target.dtype}) and source=({source.numel()}, {source.dtype})."
            )
        itemsize = int(target.itemsize)
        target_ptr = int(target.untyped_storage().data_ptr()) + int(target.storage_offset()) * itemsize
        source_ptr = int(source.untyped_storage().data_ptr()) + int(source.storage_offset()) * itemsize
        ctypes.memmove(target_ptr, source_ptr, int(target.numel()) * itemsize)

    def _storage_tensor(self, tensor: Any) -> Any:
        """Return the local storage-owning tensor for a parameter or DTensor."""
        # ParameterDTensor owns its real NPU allocation through _local_tensor.
        # Inspect it before unwrapping Parameter.data, which may expose a
        # different wrapper storage and leave the backing allocation resident.
        local_tensor = getattr(tensor, "_local_tensor", None)
        if local_tensor is not None:
            return local_tensor
        if isinstance(tensor, ms.Parameter):
            tensor = tensor.data
        local_tensor = getattr(tensor, "_local_tensor", None)
        if local_tensor is not None:
            return local_tensor
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        return tensor

    @staticmethod
    def _debug_storage_size(tensor: Any) -> Any:
        """Return storage size for diagnostics without masking runtime behavior."""
        if tensor is None:
            return None
        try:
            return tensor.untyped_storage().size()
        except (AttributeError, RuntimeError):
            return "error"

    @staticmethod
    def _align_bytes(num_bytes: int) -> int:
        return ((num_bytes + _PACKED_ALIGNMENT_BYTES - 1) // _PACKED_ALIGNMENT_BYTES) * _PACKED_ALIGNMENT_BYTES


class MindSporeSwapOptimizer:
    """MindSpore callable optimizer wrapper for state swap."""

    _is_swap_optimizer = True
    _adapters = (MindSporeNativeAdamAdapter, MindSporeNativeAdamWAdapter, MindFormersAdamWAdapter)

    def __init__(self, optimizer: Any, config: Any) -> None:
        self.optimizer = optimizer
        self.config = config
        self.runtime = MindSporeSwapRuntime(config)
        self.adapter = self._build_adapter()
        self.adapter.validate()
        # MindSpore Adam states already exist at optimizer construction, so move them off device
        # before the first forward/backward peak and let the first optimizer update prefetch them.
        initial_slots = tuple(self.adapter.initial_slots())
        self.runtime.offload_initial_slots(initial_slots)
        if self.runtime.packed_enabled:
            layout_units = self.adapter.packed_layout_units()
            layout_batches = self.runtime.partition(layout_units)
            self.runtime.prepare_packed_host(layout_batches)
        self.adapter.publish_packed_state()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the base optimizer."""
        return getattr(self.optimizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call construct for MindSpore optimizer compatibility."""
        return self.construct(*args, **kwargs)

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        """Run one optimizer update with pipeline state swap."""
        with _no_grad():
            step_context = self.adapter.prepare_step(*args, **kwargs)
            units = self.adapter.iter_update_units(step_context)
            batches = self.runtime.partition(units)
            if self.runtime.packed_enabled and self.runtime.prepare_packed_host(batches):
                self.adapter.publish_packed_state()
            result = self.runtime.run_pipeline(batches, step_context, self.adapter.step_batch)
            self.adapter.finish_step(step_context)
        return tuple(result)

    def state_dict(self) -> Dict[str, Any]:
        """Return checkpoint-safe state dict using CPU mirrors for swappable tensors."""
        self.runtime.synchronize_cpu_mirrors(self.adapter.all_slots())
        return self.adapter.checkpoint_state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load checkpoint-safe state dict while keeping swappable tensors on CPU mirrors."""
        self.adapter.load_checkpoint_state_dict(state_dict)

    def _build_adapter(self):
        for adapter_cls in self._adapters:
            if adapter_cls.matches(self.optimizer):
                return adapter_cls(self.optimizer, self.config, self.runtime)
        raise ValueError(
            "Swap optimizer only supports mindspore.nn.Adam, mindspore.nn.AdamWeightDecay "
            "(and compatible nn.AdamW aliases), and mindformers.pynative.optimizer.adamw.AdamW. "
            f"Got {type(self.optimizer)!r}."
        )


def get_swap_optimizer():
    """Return the MindSpore optimizer-state swap wrapper class."""
    return MindSporeSwapOptimizer


def _is_float_dtype(dtype: Any) -> bool:
    text = str(dtype).lower()
    return "float" in text or "bfloat" in text


def _is_cpu_tensor(tensor: Any) -> bool:
    device = getattr(tensor, "device", None)
    if device is None:
        return False
    # MindSpore may render a host device as ``CPU`` or ``CPU:0`` depending
    # on the backend/version.  Both identify host memory for raw memcpy.
    return str(device).strip().lower().split(":", 1)[0] == "cpu"
