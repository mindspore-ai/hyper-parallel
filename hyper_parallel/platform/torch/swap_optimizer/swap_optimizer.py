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
"""Torch optimizer state swap wrapper."""
# pylint: disable=protected-access

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from hyper_parallel.core.optimizer.swap_optimizer_base import PipelineSwapRuntime, SwapSlot
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.torch.swap_optimizer.adapters import (
    TorchHyperAdamWAdapter,
    TorchNativeAdamAdapter,
    TorchNativeAdamWAdapter,
)


_PACKED_ALIGNMENT_BYTES = 512
platform = get_platform()


@dataclass
class _PackedBatchRegion:
    """One dtype-contiguous host range transferred for a pipeline batch."""

    dtype: Any
    host_offset: int
    numel: int
    slots: List[SwapSlot]


@dataclass
class _PackedBatchPlan:
    """Packed transfer regions for one optimizer pipeline batch."""

    regions: Dict[Any, _PackedBatchRegion] = field(default_factory=dict)


@dataclass
class _StagingArena:
    """One raw device allocation and its dtype-specific views."""

    raw_buffer: Any
    dtype_views: Dict[Any, Any] = field(default_factory=dict)
    layout_signature: Any = None


class TorchSwapRuntime(PipelineSwapRuntime):
    """Torch tensor storage/copy runtime."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._packed_enabled = bool(getattr(config, "packed_swap", True))
        self._host_buffers: Dict[Any, Any] = {}
        self._host_layout_signature = ()
        self._packed_batch_plans: List[_PackedBatchPlan] = []
        self._staging_arenas: List[Optional[_StagingArena]] = [None, None]
        self._packed_ready_events: Dict[int, Any] = {}
        self._packed_offload_events: Dict[int, Any] = {}
        self._packed_tail_event: Optional[Any] = None
        self._packed_device_views: Dict[Any, Any] = {}

    @property
    def packed_enabled(self) -> bool:
        """Return whether this runtime may build packed state candidates."""
        return self._packed_enabled

    def populate_slot_metadata(self, slot: SwapSlot, template: Any) -> None:
        """Populate stable logical tensor metadata without allocating device state."""
        storage_tensor = self._storage_tensor(template)
        slot.shape = tuple(storage_tensor.shape)
        slot.dtype = storage_tensor.dtype
        slot.device = storage_tensor.device
        slot.numel = int(storage_tensor.numel())
        slot.storage_nbytes = slot.numel * int(storage_tensor.element_size())

    def is_packable_template(self, tensor: Any, min_numel: int) -> bool:
        """Return whether a state shaped like ``tensor`` can use packed staging."""
        if not self._packed_enabled:
            return False
        return self.is_swappable_tensor(tensor, min_numel)

    @staticmethod
    def is_distributed_tensor(tensor: Any) -> bool:
        """Return whether ``tensor`` exposes a DTensor local shard."""
        return tensor is not None and callable(getattr(tensor, "to_local", None))

    def prepare_packed_host(self, slots: Sequence[SwapSlot]) -> None:
        """Pack logical optimizer states into persistent pinned buffers by dtype."""
        if not self._packed_enabled:
            return
        packed_slots = [slot for slot in slots if slot.swappable and slot.packed]
        signature = tuple((id(slot), slot.dtype, slot.numel) for slot in packed_slots)
        if signature == self._host_layout_signature:
            return

        slots_by_dtype: Dict[Any, List[SwapSlot]] = {}
        for slot in packed_slots:
            slots_by_dtype.setdefault(slot.dtype, []).append(slot)

        new_buffers: Dict[Any, Any] = {}
        for dtype, dtype_slots in slots_by_dtype.items():
            total_numel = sum(slot.numel for slot in dtype_slots)
            host_buffer = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=True)
            new_buffers[dtype] = host_buffer
            host_offset = 0
            for slot in dtype_slots:
                flat_view = host_buffer.narrow(0, host_offset, slot.numel)
                host_view = flat_view.view(slot.shape)
                source = slot.cpu_tensor if slot.cpu_tensor is not None else slot.tensor
                if source is None:
                    host_view.zero_()
                else:
                    source_tensor = self._storage_tensor(source)
                    host_view.copy_(source_tensor.detach().reshape(-1).view(slot.shape), non_blocking=False)
                    if source_tensor.device.type != "cpu" and source is slot.tensor:
                        self.release_device_storage(slot)
                slot.host_offset = host_offset
                slot.cpu_tensor = host_view
                slot.bind_tensor(host_view)
                slot.state = "host"
                slot.event = None
                host_offset += slot.numel

        self._host_buffers = new_buffers
        self._host_layout_signature = signature

    def is_swappable_tensor(self, tensor: Any, min_numel: int) -> bool:
        """Return whether ``tensor`` can participate in swap."""
        storage_tensor = self._storage_tensor(tensor)
        if not isinstance(storage_tensor, torch.Tensor):
            return False
        if not storage_tensor.is_floating_point():
            return False
        if int(storage_tensor.numel()) < int(min_numel):
            return False
        if storage_tensor.is_sparse:
            return False
        if not storage_tensor.is_contiguous():
            return False
        if storage_tensor.device.type == "cpu":
            return False
        try:
            storage_size = int(storage_tensor.untyped_storage().size())
            expected_size = int(storage_tensor.numel()) * int(storage_tensor.element_size())
            if storage_size != expected_size:
                return False
        except RuntimeError:
            return False
        return True

    def storage_nbytes(self, tensor: Any) -> int:
        """Return storage bytes for a Torch tensor."""
        storage_tensor = self._storage_tensor(tensor)
        if not isinstance(storage_tensor, torch.Tensor):
            return 0
        try:
            return int(storage_tensor.untyped_storage().size())
        except RuntimeError:
            return int(storage_tensor.numel()) * int(storage_tensor.element_size())

    def make_cpu_tensor(self, tensor: Any) -> Any:
        """Create a CPU mirror tensor."""
        storage_tensor = self._storage_tensor(tensor)
        if isinstance(storage_tensor, torch.Tensor):
            source = storage_tensor.detach()
            try:
                cpu_tensor = torch.empty_like(source, device="cpu", pin_memory=True)
            except RuntimeError:
                cpu_tensor = torch.empty_like(source, device="cpu")
            cpu_tensor.copy_(source, non_blocking=True)
            return cpu_tensor
        raise ValueError(f"Expected torch.Tensor for CPU mirror, got {type(tensor)!r}.")

    def make_zero_cpu_tensor_like(self, tensor: Any) -> Any:
        """Create a zero-valued CPU mirror without materializing device state."""
        storage_tensor = self._storage_tensor(tensor)
        if not isinstance(storage_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for CPU mirror, got {type(tensor)!r}.")
        try:
            cpu_tensor = torch.empty_like(storage_tensor, device="cpu", pin_memory=True)
        except RuntimeError:
            cpu_tensor = torch.empty_like(storage_tensor, device="cpu")
        cpu_tensor.zero_()
        return cpu_tensor

    def make_device_tensor_like(self, param: Any, saved_tensor: Any) -> Any:
        """Create a live state tensor on the parameter device."""
        if not isinstance(saved_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in optimizer state, got {type(saved_tensor)!r}.")
        return saved_tensor.detach().to(device=param.device, dtype=saved_tensor.dtype).clone()

    def make_empty_device_tensor_like(self, param: Any, saved_tensor: Any) -> Any:
        """Create an uninitialized live state tensor shell on the parameter device."""
        if not isinstance(saved_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in optimizer state, got {type(saved_tensor)!r}.")
        return torch.empty_like(saved_tensor, device=param.device, dtype=saved_tensor.dtype)

    def copy_to_device(self, slot: SwapSlot) -> None:
        """Copy one CPU mirror to device tensor."""
        if slot.state != "host":
            return
        if slot.cpu_tensor is None:
            return
        self._storage_tensor(slot.tensor).copy_(slot.cpu_tensor, non_blocking=True)
        slot.state = "h2d"

    def wait_prefetch_slot(self, slot: SwapSlot) -> None:
        """Torch fallback copies are synchronous on CPU and stream-ordered on device."""
        slot.state = "device"

    def copy_to_cpu(self, slot: SwapSlot) -> None:
        """Copy one device tensor to CPU mirror."""
        source = self._storage_tensor(slot.tensor)
        if slot.cpu_tensor is None:
            slot.cpu_tensor = self.make_cpu_tensor(source)
        else:
            slot.cpu_tensor.copy_(source.detach(), non_blocking=True)
        slot.state = "d2h"

    def wait_offload_slot(self, slot: SwapSlot) -> None:
        """Release device storage after D2H copy completes."""
        if self._storage_tensor(slot.tensor).device.type != "cpu":
            self.release_device_storage(slot)
        slot.state = "host"

    def restore_device_storage(self, slot: SwapSlot) -> None:
        """Restore device tensor storage before H2D."""
        storage_tensor = self._storage_tensor(slot.tensor)
        if storage_tensor.device.type == "cpu":
            return
        storage = storage_tensor.untyped_storage()
        if storage.size() != slot.storage_nbytes:
            storage.resize_(slot.storage_nbytes)

    def release_device_storage(self, slot: SwapSlot) -> None:
        """Release device storage for a swappable tensor."""
        storage_tensor = self._storage_tensor(slot.tensor)
        if storage_tensor.device.type == "cpu":
            return
        storage = storage_tensor.untyped_storage()
        if storage.size() != 0:
            storage.resize_(0)

    def current_stream(self) -> Any:
        """Return the current compute stream."""
        return platform.get_current_stream()

    def new_stream(self) -> Any:
        """Create the copy stream."""
        return platform.new_stream()

    def stream_context(self, stream: Any):
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

    def supports_packed_pipeline(self, batches: Sequence[Sequence[Any]]) -> bool:
        """Make the final, step-specific decision to use the packed pipeline."""
        if not self._packed_enabled or not batches:
            return False
        swappable_slots = [
            slot
            for batch in batches
            for unit in batch
            for slot in unit.slots
            if slot.swappable
        ]
        if not swappable_slots:
            return False
        devices = {slot.device for slot in swappable_slots}
        eligible = len(devices) == 1 and None not in devices and all(
            slot.packed
            and slot.cpu_tensor is not None
            and slot.dtype in self._host_buffers
            for slot in swappable_slots
        )
        if not eligible and any(
                slot.packed and slot.logical_tensor is not None
                for slot in swappable_slots
        ):
            raise RuntimeError(
                "Packed DTensor optimizer states cannot fall back to per-tensor swap after "
                f"host packing; packed pipeline eligibility failed for local devices {sorted(map(str, devices))}."
            )
        return eligible

    @staticmethod
    def _storage_tensor(tensor: Any) -> Any:
        """Return the local tensor whose storage is managed by this runtime."""
        to_local = getattr(tensor, "to_local", None)
        return to_local() if callable(to_local) else tensor

    def begin_packed_step(self, batches: Sequence[Sequence[Any]]) -> None:
        """Build batch transfer plans and materialize two raw staging buffers."""
        first_slot = next(
            slot
            for batch in batches
            for unit in batch
            for slot in unit.slots
            if slot.swappable
        )
        # FSDP may leave gradient reductions/reshards queued on auxiliary
        # streams.  A current-stream event cannot cover those streams, so the
        # device-wide synchronization is required before optimizer reads state.
        getattr(torch, first_slot.device.type).synchronize(first_slot.device)
        self._packed_tail_event = None
        self._packed_batch_plans = [self._build_packed_batch_plan(batch) for batch in batches]
        max_numel_by_dtype: Dict[Any, int] = {}
        device = None
        for batch_plan in self._packed_batch_plans:
            for dtype, region in batch_plan.regions.items():
                max_numel_by_dtype[dtype] = max(max_numel_by_dtype.get(dtype, 0), region.numel)
                if device is None and region.slots:
                    device = region.slots[0].device
        if device is None:
            raise RuntimeError("Packed optimizer pipeline has no device-resident state metadata.")

        dtype_layouts = {}
        byte_offset = 0
        for dtype in sorted(max_numel_by_dtype, key=str):
            byte_offset = self._align_bytes(byte_offset)
            element_size = int(self._host_buffers[dtype].element_size())
            num_bytes = max_numel_by_dtype[dtype] * element_size
            dtype_layouts[dtype] = (byte_offset, num_bytes)
            byte_offset += num_bytes
        total_bytes = self._align_bytes(byte_offset)

        for staging_index in range(2):
            arena = self._materialize_staging_arena(staging_index, total_bytes, device)
            layout_signature = tuple(
                (dtype, offset, num_bytes) for dtype, (offset, num_bytes) in dtype_layouts.items()
            )
            if arena.layout_signature != layout_signature:
                arena.dtype_views = {
                    dtype: arena.raw_buffer.narrow(0, offset, num_bytes).view(dtype)
                    for dtype, (offset, num_bytes) in dtype_layouts.items()
                }
                arena.layout_signature = layout_signature
                self._drop_packed_views(staging_index)
        self._packed_ready_events = {}
        self._packed_offload_events = {}

    def enqueue_packed_prefetch(self, batch_index: int, staging_index: int) -> None:
        """Enqueue one packed H2D and record its ready event."""
        copy_stream = self._get_copy_stream()
        with self.stream_context(copy_stream):
            self._copy_packed_to_device(batch_index, staging_index)
            ready_event = self.record_event(copy_stream)
        self._packed_ready_events[batch_index] = ready_event
        self._packed_tail_event = ready_event

    def wait_packed_prefetch(self, batch_index: int, staging_index: int) -> None:
        """Order the compute stream after the batch's packed transfer chain."""
        del staging_index
        ready_event = self._packed_ready_events.get(batch_index)
        if ready_event is None:
            raise RuntimeError(f"Packed optimizer batch {batch_index} has no ready event.")
        self.wait_event(ready_event, self.current_stream())

    def activate_packed_batch(self, batch_index: int, staging_index: int) -> None:
        """Bind each swap slot to its slice of one staging arena."""
        batch_plan = self._packed_batch_plans[batch_index]
        arena = self._require_staging_arena(staging_index)
        for dtype, region in batch_plan.regions.items():
            dtype_view = arena.dtype_views[dtype]
            for slot in region.slots:
                relative_offset = slot.host_offset - region.host_offset
                cache_key = (
                    staging_index,
                    id(arena.raw_buffer),
                    id(dtype_view),
                    id(slot),
                    dtype,
                    relative_offset,
                    slot.numel,
                    slot.shape,
                )
                device_view = self._packed_device_views.get(cache_key)
                if device_view is None:
                    device_view = dtype_view.narrow(0, relative_offset, slot.numel).view(slot.shape)
                    self._packed_device_views[cache_key] = device_view
                slot.bind_tensor(device_view)
                slot.state = "device"
                slot.event = None

    def enqueue_packed_offload_prefetch(
            self,
            batch_index: int,
            next_index: Optional[int],
            staging_index: int,
    ) -> None:
        """Serialize current D2H and next same-parity H2D on the copy stream."""
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
        """Order the compute stream after a packed transfer chain."""
        offload_event = self._packed_offload_events.get(batch_index)
        if offload_event is None:
            raise RuntimeError(f"Packed optimizer batch {batch_index} has no offload event.")
        self.wait_event(offload_event, self.current_stream())

    def finish_packed_offload(self, batch_index: int) -> None:
        """Make persistent pinned views authoritative after D2H completion."""
        batch_plan = self._packed_batch_plans[batch_index]
        for region in batch_plan.regions.values():
            for slot in region.slots:
                slot.bind_tensor(slot.cpu_tensor)
                slot.state = "host"
                slot.event = None

    def end_packed_step(self) -> None:
        """Drain the packed copy chain and release step-local staging storage."""
        if self._packed_tail_event is not None:
            # All D2H copies are serialized on the copy stream.  Waiting only
            # for its tail is sufficient before detaching views and releasing
            # the step-local device allocation.
            self.wait_event(self._packed_tail_event, None)
        active_slots = {
            id(slot): slot
            for batch_plan in self._packed_batch_plans
            for region in batch_plan.regions.values()
            for slot in region.slots
            if slot.state == "device"
        }
        if active_slots:
            compute_stream = self.current_stream()
            synchronize = getattr(compute_stream, "synchronize", None)
            if synchronize is not None:
                synchronize()
            for slot in active_slots.values():
                slot.cpu_tensor.copy_(self._storage_tensor(slot.tensor).detach(), non_blocking=False)
                slot.bind_tensor(slot.cpu_tensor)
                slot.state = "host"
                slot.event = None
        for arena in self._staging_arenas:
            if arena is None:
                continue
            # Drop views before shrinking the raw storage; otherwise a view
            # can keep the device allocation alive after the step ends.
            arena.dtype_views = {}
            arena.layout_signature = None
            storage = arena.raw_buffer.untyped_storage()
            if storage.size() != 0:
                storage.resize_(0)
        self._packed_device_views = {}
        self._packed_batch_plans = []
        self._packed_ready_events = {}
        self._packed_offload_events = {}
        self._packed_tail_event = None

    def _build_packed_batch_plan(self, batch: Sequence[Any]) -> _PackedBatchPlan:
        """Group a batch's packed slots into contiguous host regions by dtype."""
        slots_by_dtype: Dict[Any, List[SwapSlot]] = {}
        seen_slots = set()
        for unit in batch:
            for slot in unit.slots:
                if not slot.swappable or not slot.packed or id(slot) in seen_slots:
                    continue
                slots_by_dtype.setdefault(slot.dtype, []).append(slot)
                seen_slots.add(id(slot))

        regions = {}
        for dtype, slots in slots_by_dtype.items():
            slots.sort(key=lambda slot: slot.host_offset)
            host_offset = slots[0].host_offset
            expected_offset = host_offset
            for slot in slots:
                if slot.host_offset != expected_offset:
                    raise RuntimeError(
                        f"Packed optimizer batch has a non-contiguous {dtype} host range at slot {slot.name!r}."
                    )
                expected_offset += slot.numel
            regions[dtype] = _PackedBatchRegion(dtype, host_offset, expected_offset - host_offset, slots)
        return _PackedBatchPlan(regions)

    def _materialize_staging_arena(self, staging_index: int, total_bytes: int, device: Any) -> _StagingArena:
        """Allocate or resize one packed device staging arena for the requested layout."""
        arena = self._staging_arenas[staging_index]
        if arena is None or arena.raw_buffer.device != device:
            raw_buffer = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            arena = _StagingArena(raw_buffer)
            self._staging_arenas[staging_index] = arena
            self._drop_packed_views(staging_index)
            return arena

        raw_buffer = arena.raw_buffer
        storage = raw_buffer.untyped_storage()
        if storage.size() < total_bytes:
            storage.resize_(total_bytes)
            arena.dtype_views = {}
            arena.layout_signature = None
            self._drop_packed_views(staging_index)
        raw_buffer.set_(storage, 0, (total_bytes,), (1,))
        return arena

    def _drop_packed_views(self, staging_index: int) -> None:
        """Drop cached views for one arena after its storage/layout changes."""
        self._packed_device_views = {
            key: value for key, value in self._packed_device_views.items() if key[0] != staging_index
        }

    def _copy_packed_to_device(self, batch_index: int, staging_index: int) -> None:
        batch_plan = self._packed_batch_plans[batch_index]
        arena = self._require_staging_arena(staging_index)
        for dtype, region in batch_plan.regions.items():
            host_view = self._host_buffers[dtype].narrow(0, region.host_offset, region.numel)
            arena.dtype_views[dtype].narrow(0, 0, region.numel).copy_(host_view, non_blocking=True)

    def _copy_packed_to_host(self, batch_index: int, staging_index: int) -> None:
        batch_plan = self._packed_batch_plans[batch_index]
        arena = self._require_staging_arena(staging_index)
        for dtype, region in batch_plan.regions.items():
            host_view = self._host_buffers[dtype].narrow(0, region.host_offset, region.numel)
            host_view.copy_(arena.dtype_views[dtype].narrow(0, 0, region.numel), non_blocking=True)

    def _require_staging_arena(self, staging_index: int) -> _StagingArena:
        arena = self._staging_arenas[staging_index]
        if arena is None:
            raise RuntimeError(f"Packed optimizer staging arena {staging_index} is not materialized.")
        return arena

    @staticmethod
    def _align_bytes(num_bytes: int) -> int:
        return ((num_bytes + _PACKED_ALIGNMENT_BYTES - 1) // _PACKED_ALIGNMENT_BYTES) * _PACKED_ALIGNMENT_BYTES


class TorchSwapOptimizer(torch.optim.Optimizer):
    """Torch optimizer wrapper for Adam/AdamW state swap."""

    _is_swap_optimizer = True
    _adapters = (TorchHyperAdamWAdapter, TorchNativeAdamAdapter, TorchNativeAdamWAdapter)

    def __init__(self, optimizer: Any, config: Any) -> None:
        # Do not call ``torch.optim.Optimizer.__init__``: the wrapped base
        # optimizer already owns param_groups/state/defaults. Inheriting keeps
        # PyTorch LR schedulers and isinstance checks happy while this wrapper
        # delegates all optimizer state to ``self.optimizer``.
        self.optimizer = optimizer
        self.config = config
        self.runtime = TorchSwapRuntime(config)
        self.adapter = self._build_adapter()
        self.adapter.validate()
        # Torch Adam states are normally lazy, but callers may materialize them
        # before wrapping to avoid first-step initialization in the measured loop.
        initial_slots = tuple(self.adapter.initial_slots())
        self.runtime.offload_initial_slots(initial_slots)
        self.runtime.prepare_packed_host(initial_slots)
        self.adapter.publish_packed_state()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the base optimizer."""
        return getattr(self.optimizer, name)

    @property
    def param_groups(self):
        """Proxy parameter groups."""
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value) -> None:
        self.optimizer.param_groups = value

    @property
    def state(self):
        """Proxy optimizer state."""
        return self.optimizer.state

    @property
    def defaults(self):
        """Proxy optimizer defaults."""
        return self.optimizer.defaults

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Proxy param group addition."""
        self.optimizer.add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Proxy gradient clearing."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Any] = None) -> Any:
        """Run one optimizer step with pipeline state swap."""
        if closure is not None:
            raise ValueError("Swap optimizer does not support closure.")
        with self._no_grad_context():
            step_context = self.adapter.prepare_step()
            units = self.adapter.iter_update_units(step_context)
            batches = self.runtime.partition(units)
            self.runtime.run_pipeline(batches, step_context, self.adapter.step_batch)
            return self.adapter.finish_step(step_context)

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dict using CPU mirrors for swappable tensors."""
        return self.adapter.checkpoint_state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dict while keeping swappable tensors on CPU mirrors."""
        self.adapter.load_checkpoint_state_dict(state_dict)

    def _build_adapter(self):
        for adapter_cls in self._adapters:
            if adapter_cls.matches(self.optimizer):
                return adapter_cls(self.optimizer, self.config, self.runtime)
        raise ValueError(
            "Swap optimizer only supports torch.optim.Adam, torch.optim.AdamW, "
            "and hyper_parallel.core.optimizer.adamw.AdamW on the Torch backend. "
            f"Got {type(self.optimizer)!r}."
        )

    @contextlib.contextmanager
    def _no_grad_context(self) -> Iterable[None]:
        with torch.no_grad():
            yield


def get_swap_optimizer():
    """Return the Torch optimizer-state swap wrapper class."""
    return TorchSwapOptimizer
