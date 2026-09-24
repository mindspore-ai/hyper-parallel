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
"""Generic optimizer-state swap runtime, slots and adapter base."""
# pylint: disable=protected-access

from __future__ import annotations

import contextlib
import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch

# Logical optimizer-state keys every swap family in this codebase understands.
# This is the vocabulary ``SwapOptimizerConfig.state_keys`` may name; each
# adapter still validates that a key belongs to its own optimizer family.
KNOWN_STATE_KEYS = (
    "exp_avg",
    "exp_avg_sq",
    "max_exp_avg_sq",
    "momentum_buffer",
    "master_param",
)
_PACKED_ALIGNMENT_BYTES = 512
_DEVICE_TYPE = "npu"


@dataclass
class SwapSlot:
    """One logical optimizer state tensor that may be swapped."""

    name: str
    tensor: Any
    cpu_tensor: Optional[Any] = None
    storage_nbytes: int = 0
    swappable: bool = True
    state: str = "device"
    event: Optional[Any] = None
    shape: tuple[int, ...] = ()
    dtype: Optional[Any] = None
    device: Optional[Any] = None
    numel: int = 0
    host_offset: int = 0
    packed: bool = False
    logical_tensor: Optional[Any] = None

    def bind_tensor(self, tensor: Any) -> None:
        """Bind the slot to a host or staging tensor view."""
        device = getattr(tensor, "device", None)
        device_type = getattr(device, "type", None)
        if self.logical_tensor is not None and device_type != "cpu":
            # DTensor exposes no public setter for replacing its local shard.
            setattr(self.logical_tensor, "_local_tensor", tensor)
            self.logical_tensor.data = tensor
            self.tensor = self.logical_tensor
        else:
            self.tensor = tensor


    @property
    def checkpoint_tensor(self) -> Any:
        """Return the CPU tensor when present, otherwise the live tensor."""
        return self.cpu_tensor if self.cpu_tensor is not None else self.tensor


class SwapUpdateUnit(Protocol):
    """Minimal contract the pipeline runtime requires of an update unit.

    The runtime only ever reads a unit's swappable slots, so an update unit is
    free to carry algorithm-specific metadata (a single parameter and gradient,
    a Muon HSDP assignment, collective ordering, and so on) as long as it
    exposes ``slots``.
    """

    slots: List[SwapSlot]


@dataclass
class UpdateUnit:
    """Generic per-parameter update unit: parameter, gradient and swap slots.

    Kept as a concrete unit for optimizers whose atomic update is one parameter,
    and as the runtime's default unit type.  ``adapter_index`` identifies the
    owning parameter group.  Optimizers with a richer atomic unit (Muon HSDP
    assignments, for example) define their own unit exposing ``slots``.
    """

    adapter_index: int
    param: Any
    grad: Any
    slots: List[SwapSlot]


class PipelineSwapRuntime:
    """Torch tensor storage/copy runtime and pipeline orchestration."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._copy_stream: Optional[Any] = None
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

    def partition(self, units: Sequence[UpdateUnit]) -> List[List[UpdateUnit]]:
        """Partition update units into balanced batches by swappable state bytes."""
        non_empty_units = list(units)
        if not non_empty_units:
            return []
        swap_times = max(1, min(int(self.config.swap_times), len(non_empty_units)))
        unit_costs = [max(1, self._unit_cost(unit)) for unit in non_empty_units]
        remaining_cost = sum(unit_costs)
        batches: List[List[UpdateUnit]] = []
        start = 0
        for batch_index in range(swap_times - 1):
            remaining_batches = swap_times - batch_index

            # Compare against the remaining average without introducing floating-point rounding.
            # Bind the loop state as defaults so the closure captures this iteration's values.
            def still_gaining(
                    end: int,
                    current_cost: int,
                    next_cost: int,
                    remaining_batches: int = remaining_batches,
                    remaining_cost: int = remaining_cost,
                    start: int = start,
            ) -> bool:
                current_distance = abs(remaining_cost - current_cost * remaining_batches)
                next_distance = abs(remaining_cost - (current_cost + next_cost) * remaining_batches)
                return end > start and current_distance <= next_distance

            max_end = len(non_empty_units) - remaining_batches + 1
            end, current_cost = self._grow_balanced_segment(unit_costs, start, max_end, still_gaining)
            batches.append(non_empty_units[start:end])
            remaining_cost -= current_cost
            start = end
        batches.append(non_empty_units[start:])
        return batches

    def run_pipeline(
            self,
            batches: Sequence[Sequence[UpdateUnit]],
            step_context: Any,
            step_batch: Callable[[List[UpdateUnit], Any], Any],
    ) -> List[Any]:
        """Run one-batch-ahead prefetch while releasing completed offloads before widening the window."""
        results = []
        batch_lists = [list(batch) for batch in batches]
        if not batch_lists:
            return results

        if self.supports_packed_pipeline(batch_lists):
            return self._run_packed_pipeline(batch_lists, step_context, step_batch)

        try:
            self.prefetch(batch_lists[0]) # prefetch 0
            for index, batch_list in enumerate(batch_lists):
                self.wait_prefetch(batch_list) # wait_prefetch n

                previous_index = index - 1
                if previous_index >= 0:
                    self.wait_offload(batch_lists[previous_index]) # wait_offload n-1

                next_index = index + 1
                if next_index < len(batch_lists):
                    self.prefetch(batch_lists[next_index]) # prefetch n+1

                results.append(step_batch(batch_list, step_context)) # update n
                self.refresh_swappable_slots(batch_list)
                self.offload(batch_list) # offload n

            self.wait_offload(batch_lists[-1])
        finally:
            self._drain_pending_transfers(batch_lists)
        return results

    def _drain_pending_transfers(self, batch_lists: Sequence[Sequence[UpdateUnit]]) -> None:
        """Settle slots left in an intermediate ``h2d``/``d2h`` state.

        Called from ``run_pipeline``'s ``finally`` so that an exception raised
        mid-pipeline (for example in ``step_batch``) cannot leave prefetched or
        offloaded slots holding device storage and a pending stream event. Slots
        that already reached a terminal state (``device`` or ``host``) are left
        untouched.
        """
        for batch_list in batch_lists:
            self.wait_prefetch(batch_list)
            self.wait_offload(batch_list)

    def _run_packed_pipeline(
            self,
            batches: Sequence[List[UpdateUnit]],
            step_context: Any,
            step_batch: Callable[[List[UpdateUnit], Any], Any],
    ) -> List[Any]:
        """Run updates with two reusable staging buffers.

        Buffer parity is fixed by batch index. D2H batch ``n`` and H2D batch
        ``n + 2`` share one copy-stream chain while the other arena updates.
        """
        results = []
        try:
            self.begin_packed_step(batches)
            self.enqueue_packed_prefetch(0, 0)
            if len(batches) > 1:
                self.enqueue_packed_prefetch(1, 1)
            for batch_index, batch in enumerate(batches):
                staging_index = batch_index % 2
                self.wait_packed_prefetch(batch_index, staging_index)
                completed_index = batch_index - 2
                if completed_index >= 0:
                    self.wait_packed_offload(completed_index)
                    self.finish_packed_offload(completed_index)
                self.activate_packed_batch(batch_index, staging_index)
                results.append(step_batch(batch, step_context))
                self.refresh_swappable_slots(batch)
                next_index = batch_index + 2
                self.enqueue_packed_offload_prefetch(
                    batch_index,
                    next_index if next_index < len(batches) else None,
                    staging_index,
                )
            drain_start = max(0, len(batches) - 2)
            for batch_index in range(drain_start, len(batches)):
                self.wait_packed_offload(batch_index)
                self.finish_packed_offload(batch_index)
        finally:
            self.release_packed_step_results(results)
            self.end_packed_step()
        return results

    # Both hooks below are dispatched through ``self`` from the packed pipeline
    # above and are overridden per backend, so they stay instance
    # methods even though the base bodies only discard their arguments.
    def release_packed_step_results(self, results: List[Any]) -> None:
        """Release backend-specific update outputs before staging teardown."""
        del results

    def refresh_swappable_slots(self, batch: Sequence[UpdateUnit]) -> None:
        """Refresh slots that become swappable after the optimizer update."""
        del batch

    def synchronize_cpu_mirrors(self, slots: Iterable[SwapSlot]) -> None:
        """Ensure CPU mirrors contain latest data for checkpointing."""
        slot_list = [slot for slot in _iter_unique_slot_objects(slots) if slot.swappable]
        if not slot_list:
            return

        compute_stream = self.current_stream()
        with self.stream_context(compute_stream):
            for event in _iter_unique_events(slot_list):
                self.wait_event(event, compute_stream)

            for slot in slot_list:
                if slot.state == "host":
                    if slot.cpu_tensor is None:
                        raise RuntimeError(f"Swap slot {slot.name!r} is host-resident but has no CPU mirror.")
                    continue
                if slot.state != "d2h":
                    self.copy_to_cpu(slot)
                self.wait_offload_slot(slot)

            checkpoint_event = self.record_event(compute_stream) if compute_stream is not None else None

        # Storage release only needs stream ordering above. Host completion is
        # required separately because checkpoint_state_dict reads CPU mirrors.
        self.wait_event(checkpoint_event, None)
        for slot in slot_list:
            slot.event = None

    def prefetch(self, batch: Sequence[UpdateUnit]) -> None:
        """Prefetch batch slots from CPU to device."""
        slots = [slot for slot in _iter_unique_slots(batch) if slot.swappable and slot.state == "host"]
        if not slots:
            return

        for slot in slots:
            if slot.cpu_tensor is None:
                raise RuntimeError(f"Swap slot {slot.name!r} is host-resident but has no CPU mirror.")

        for slot in slots:
            self.restore_device_storage(slot)

        copy_stream = self._get_copy_stream()
        compute_event = self._record_current_stream_event() if copy_stream is not None else None
        with self.stream_context(copy_stream):
            self.wait_event(compute_event, copy_stream)
            for slot in slots:
                self.copy_to_device(slot)
                slot.state = "h2d"
            copy_event = self.record_event(copy_stream) if copy_stream is not None else None
        for slot in slots:
            slot.event = copy_event

    def wait_prefetch(self, batch: Sequence[UpdateUnit]) -> None:
        """Wait for batch prefetch copies."""
        slots = [slot for slot in _iter_unique_slots(batch) if slot.swappable and slot.state == "h2d"]
        if not slots:
            return

        compute_stream = self.current_stream()
        with self.stream_context(compute_stream):
            for event in _iter_unique_events(slots):
                self.wait_event(event, compute_stream)
            for slot in slots:
                self.wait_prefetch_slot(slot)
                slot.event = None

    def offload(self, batch: Sequence[UpdateUnit]) -> None:
        """Offload batch slots from device to CPU."""
        slots = [slot for slot in _iter_unique_slots(batch) if slot.swappable and slot.state == "device"]
        if not slots:
            return

        self._enqueue_offload_slots(slots)

    def offload_initial_slots(self, slots: Iterable[SwapSlot]) -> None:
        """Offload existing device-resident slots before the first optimizer update."""
        slot_list = [
            slot for slot in _iter_unique_slot_objects(slots)
            if slot.swappable and slot.state == "device"
        ]
        if not slot_list:
            return

        self._enqueue_offload_slots(slot_list)
        # Waiting here also releases device storage. Deferring this until the first
        # prefetch would leave cold optimizer states resident during forward/backward.
        self._wait_offload_slots(slot_list)

    def _enqueue_offload_slots(self, slots: Sequence[SwapSlot]) -> None:
        """Enqueue D2H copies for device-resident slots."""
        copy_stream = self._get_copy_stream()
        compute_event = self._record_current_stream_event() if copy_stream is not None else None
        with self.stream_context(copy_stream):
            self.wait_event(compute_event, copy_stream)
            for slot in slots:
                self.copy_to_cpu(slot)
                slot.state = "d2h"
            copy_event = self.record_event(copy_stream) if copy_stream is not None else None
        for slot in slots:
            slot.event = copy_event

    def wait_offload(self, batch: Sequence[UpdateUnit]) -> None:
        """Wait for batch offload copies."""
        slots = [slot for slot in _iter_unique_slots(batch) if slot.swappable and slot.state == "d2h"]
        if not slots:
            return

        self._wait_offload_slots(slots)

    def _wait_offload_slots(self, slots: Sequence[SwapSlot]) -> None:
        """Wait for D2H copies and release device storage for copied slots."""
        compute_stream = self.current_stream()
        with self.stream_context(compute_stream):
            for event in _iter_unique_events(slots):
                self.wait_event(event, compute_stream)
            for slot in slots:
                self.wait_offload_slot(slot)
                slot.event = None

    @staticmethod
    def _unit_cost(unit: UpdateUnit) -> int:
        return sum(slot.storage_nbytes for slot in unit.slots if slot.swappable)

    @staticmethod
    def _grow_balanced_segment(
            unit_costs: Sequence[int],
            start: int,
            max_end: int,
            should_stop: Callable[[int, int, int], bool],
    ) -> Tuple[int, int]:
        """Extend a segment from ``start`` while ``should_stop`` stays false.

        Returns the exclusive end index and the accumulated cost of ``[start, end)``.
        """
        end = start
        current_cost = 0
        while end < max_end:
            next_cost = unit_costs[end]
            if should_stop(end, current_cost, next_cost):
                break
            current_cost += next_cost
            end += 1
        return end, current_cost

    def _get_copy_stream(self) -> Any:
        if self._copy_stream is None:
            self._copy_stream = self.new_stream()
        return self._copy_stream

    def _record_current_stream_event(self) -> Any:
        current_stream = self.current_stream()
        return self.record_event(current_stream)

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

    @staticmethod
    def validate_packed_devices(slots: Sequence[SwapSlot]) -> None:
        """Reject packed host packing when swappable states span multiple local devices.

        A packed slot rebinds its live tensor to a CPU host view, so once host
        packing has run the runtime can no longer fall back to per-tensor swap.
        Callers therefore validate device uniformity *before* any slot is
        rebound, rather than discovering the mismatch in
        :meth:`supports_packed_pipeline` after the optimizer state is already
        committed to host storage.

        Args:
            slots: Candidate swappable slots, before host packing.

        Raises:
            RuntimeError: If the slots do not all live on one known device.
        """
        devices = {slot.device for slot in slots}
        if len(devices) > 1 or None in devices:
            raise RuntimeError(
                "Packed optimizer states must live on a single local device before host "
                f"packing; got {sorted(map(str, devices))}. Set packed_swap=False to swap "
                "states that span devices."
            )

    def prepare_packed_host(self, slots: Sequence[SwapSlot]) -> None:
        """Pack logical optimizer states into persistent pinned buffers by dtype."""
        if not self._packed_enabled:
            return
        packed_slots = [slot for slot in slots if slot.swappable and slot.packed]
        self.validate_packed_devices(packed_slots)
        signature = tuple((id(slot), slot.dtype, slot.numel) for slot in packed_slots)
        if signature == self._host_layout_signature:
            return

        slots_by_dtype: Dict[Any, List[SwapSlot]] = {}
        for slot in packed_slots:
            slots_by_dtype.setdefault(slot.dtype, []).append(slot)

        new_buffers: Dict[Any, Any] = {}
        for dtype, dtype_slots in slots_by_dtype.items():
            new_buffers[dtype] = self._pack_dtype_slots(dtype, dtype_slots)

        self._host_buffers = new_buffers
        self._host_layout_signature = signature

    def _pack_dtype_slots(self, dtype: Any, dtype_slots: Sequence[SwapSlot]) -> Any:
        """Pack every slot of one dtype into a single pinned host buffer.

        Each slot is re-pointed at its own view inside the new buffer and marked
        host-resident, so the packed buffer becomes the slot's CPU mirror from
        here on.  Any device copy that fed the mirror is released immediately
        afterwards, because the host view now owns the only valid values.
        """
        total_numel = sum(slot.numel for slot in dtype_slots)
        host_buffer = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=True)
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
        return host_buffer

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

    @staticmethod
    def make_device_tensor_like(param: Any, saved_tensor: Any) -> Any:
        """Create a live state tensor on the parameter device."""
        if not isinstance(saved_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in optimizer state, got {type(saved_tensor)!r}.")
        return saved_tensor.detach().to(device=param.device, dtype=saved_tensor.dtype).clone()

    @staticmethod
    def make_empty_device_tensor_like(param: Any, saved_tensor: Any) -> Any:
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

    @staticmethod
    def device_handle() -> Any:
        """Return the Torch device module (``torch.npu``) that owns streams and events."""
        try:
            return getattr(torch, _DEVICE_TYPE)
        except AttributeError as error:
            raise RuntimeError(
                f"Swap optimizer expects device handle: 'torch.{_DEVICE_TYPE}' failed."
            ) from error

    def current_stream(self) -> Any:
        """Return the current compute stream."""
        return self.device_handle().current_stream()

    def new_stream(self) -> Any:
        """Create the copy stream."""
        return self.device_handle().Stream()

    def stream_context(self, stream: Any):
        """Return a stream context for ``stream``."""
        if stream is None:
            return contextlib.nullcontext()
        return self.device_handle().stream(stream)

    def record_event(self, stream: Any = None) -> Any:
        """Record an event on ``stream``."""
        event = self.device_handle().Event()
        if stream is None:
            event.record()
        else:
            event.record(stream)
        return event

    @staticmethod
    def wait_event(event: Any, stream: Any = None) -> None:
        """Make ``stream`` wait for ``event``."""
        if event is None:
            return
        if stream is None:
            event.synchronize()
            return
        event.wait(stream)

    def supports_packed_pipeline(self, batches: Sequence[Sequence[UpdateUnit]]) -> bool:
        """Make the final, step-specific decision to use the packed pipeline."""
        if not self._packed_enabled or not batches:
            return False
        units = itertools.chain.from_iterable(batches)
        swappable_slots = [
            slot for unit in units for slot in unit.slots if slot.swappable
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
        # A packed slot always carries a host mirror and a host-buffer dtype once
        # ``prepare_packed_host`` has run, so the only remaining way for
        # ``eligible`` to be false is a device mismatch.  Per-tensor swap cannot
        # recover from that -- the slot tensor is already a CPU host view -- so
        # every packed slot must fail loudly rather than fall through silently.
        if not eligible and any(slot.packed for slot in swappable_slots):
            raise RuntimeError(
                "Packed optimizer states cannot fall back to per-tensor swap after "
                f"host packing; packed pipeline eligibility failed for local devices {sorted(map(str, devices))}."
            )
        return eligible

    @staticmethod
    def _storage_tensor(tensor: Any) -> Any:
        """Return the local tensor whose storage is managed by this runtime."""
        to_local = getattr(tensor, "to_local", None)
        return to_local() if callable(to_local) else tensor

    @staticmethod
    def _first_swappable_slot(batches: Sequence[Sequence[UpdateUnit]]) -> SwapSlot:
        """Return the first swappable slot across ``batches``, in iteration order."""
        units = itertools.chain.from_iterable(batches)
        swappable = (slot for unit in units for slot in unit.slots if slot.swappable)
        return next(swappable)

    def begin_packed_step(self, batches: Sequence[Sequence[UpdateUnit]]) -> None:
        """Build batch transfer plans and materialize two raw staging buffers."""
        first_slot = self._first_swappable_slot(batches)
        # FSDP may leave gradient reductions/reshards queued on auxiliary
        # streams.  A current-stream event cannot cover those streams, so the
        # device-wide synchronization is required before optimizer reads state.
        getattr(torch, first_slot.device.type).synchronize(first_slot.device)
        self._packed_tail_event = None
        self._packed_batch_plans = [self._build_packed_batch_plan(batch) for batch in batches]
        dtype_layouts, total_bytes, device = self._packed_layout(self._packed_batch_plans)

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
        active_slots: Dict[int, Any] = {}
        for batch_plan in self._packed_batch_plans:
            for region in batch_plan.regions.values():
                for slot in region.slots:
                    if slot.state == "device":
                        active_slots[id(slot)] = slot
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

    @staticmethod
    def _build_packed_batch_plan(batch: Sequence[UpdateUnit]) -> _PackedBatchPlan:
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

    def _packed_layout(self, batch_plans: Sequence[_PackedBatchPlan]) -> Tuple[Dict[Any, Tuple[int, int]], int, Any]:
        """Return the dtype -> (offset, bytes) staging layout shared by every batch."""
        max_numel_by_dtype: Dict[Any, int] = {}
        device = None
        for batch_plan in batch_plans:
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
        return dtype_layouts, self._align_bytes(byte_offset), device


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


def _iter_unique_slots(units: Sequence[UpdateUnit]) -> Iterable[SwapSlot]:
    """Yield slots once by object identity."""
    slots = itertools.chain.from_iterable(unit.slots for unit in units)
    return _iter_unique_slot_objects(slots)


def _iter_unique_slot_objects(slots: Iterable[SwapSlot]) -> Iterable[SwapSlot]:
    """Yield slots once by tensor object identity."""
    unique_slots: Dict[int, SwapSlot] = {}
    for slot in slots:
        unique_slots.setdefault(id(slot.tensor), slot)
    return unique_slots.values()


def _iter_unique_events(slots: Iterable[SwapSlot]) -> Iterable[Any]:
    """Yield non-empty events once by object identity."""
    seen = set()
    for slot in slots:
        event = slot.event
        if event is None:
            continue
        key = id(event)
        if key in seen:
            continue
        seen.add(key)
        yield event


def _slot_tensor(unit: SwapUpdateUnit, key: str, fallback: Any) -> Any:
    """Return an active swap slot tensor for ``key``, or ``fallback``.

    A slot is authoritative only while it is swappable, still resident on the
    local device and bound to a live tensor; otherwise the optimizer state entry
    is the source of truth for this step.
    """
    for slot in unit.slots:
        if slot.name == key and slot.swappable and slot.state == "device" and slot.tensor is not None:
            return slot.tensor
    return fallback


def validate_state_keys(state_keys: Optional[Sequence[str]]) -> Optional[tuple[str, ...]]:
    """Validate user-provided optimizer state keys.

    Only lexical validity is checked here: a key must appear in
    :data:`KNOWN_STATE_KEYS`.  Whether the wrapped optimizer family actually
    owns that state is decided later by the family adapter, which fails during
    ``validate()``/wrap time.
    """
    if state_keys is None:
        return None
    normalized = tuple(state_keys)
    invalid = sorted(set(normalized) - set(KNOWN_STATE_KEYS))
    if invalid:
        raise ValueError(
            "SwapOptimizerConfig.state_keys only supports optimizer state slots "
            f"{KNOWN_STATE_KEYS}, but got {invalid}."
        )
    return normalized


#: Builds the family adapter for a wrapped optimizer.
AdapterFactory = Callable[[Any, Any, PipelineSwapRuntime], "StateSwapAdapter"]


class StateSwapAdapter:
    """Algorithm-independent optimizer-state swap adapter base.

    Owns the slot map, the configured/default state-key selection framework, the
    checkpoint strip/export/restore path and packed-state publication.  The
    numerical update and the per-parameter unit construction belong to the
    optimizer family subclasses, which implement :meth:`default_state_keys`,
    :meth:`prepare_step`, :meth:`iter_update_units` and :meth:`step_batch`.
    """

    #: Optimizer classes this adapter accepts; ``()`` means "decided by subclass".
    supported_cls = ()

    def default_state_keys(self) -> Tuple[str, ...]:
        """Return the family's default swap state keys.

        Subclasses must override this; the base class deliberately assumes
        nothing about an optimizer's state layout, so there is no default.

        Raises:
            NotImplementedError: Always, unless a subclass overrides it.
        """
        raise NotImplementedError

    def __init__(self, optimizer: Any, config: Any, runtime: Any) -> None:
        self.optimizer = optimizer
        self.config = config
        self.runtime = runtime
        self._slots: Dict[Tuple[int, str], SwapSlot] = {}

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        """Return whether this adapter supports ``optimizer``."""
        return isinstance(optimizer, cls.supported_cls)

    def iter_update_units(self, step_context: Dict[str, Any]) -> List[UpdateUnit]:
        """Return units collected in ``prepare_step``."""
        return step_context["units"]

    def initial_slots(self) -> Iterable[SwapSlot]:
        """Discover optimizer states materialized before the swap wrapper was created."""
        slots = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if state:
                    slots.extend(self._build_slots(param, state))
        return tuple(slots)

    def finish_step(self, step_context: Any) -> Any:
        """Finish one outer optimizer step."""
        del step_context

    def all_slots(self) -> Iterable[SwapSlot]:
        """Iterate known swap slots."""
        return tuple(self._ordered_slots())

    def checkpoint_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return optimizer checkpoint state using CPU mirrors for swapped slots."""
        del args, kwargs
        self.runtime.synchronize_cpu_mirrors(self.all_slots())
        return self.export_swappable_state(self.optimizer.state_dict())

    def load_checkpoint_state_dict(
            self,
            state_dict: Dict[str, Any],
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """Load optimizer checkpoint state while restoring swap-managed slots."""
        del args, kwargs
        stripped, removed = self.strip_swappable_state(state_dict)
        self.optimizer.load_state_dict(stripped)
        self.load_swappable_state(state_dict, removed)

    def publish_packed_state(self) -> None:
        """Publish persistent packed CPU mirrors to the wrapped optimizer state."""
        if not self.runtime.packed_enabled:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                for key in self._configured_state_keys():
                    slot = self._slots.get((id(param), key))
                    if slot is None or not slot.packed or slot.cpu_tensor is None:
                        continue
                    if state is None:
                        state = self.optimizer.state[param]
                    state[key] = slot.cpu_tensor

    def export_swappable_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build a checkpoint-safe Torch optimizer state dict.

        Torch optimizer state dicts are keyed by saved parameter ids, while the
        adapter tracks live swap slots by the current parameter objects.  This
        method walks both orders together and exports each parameter's optimizer
        state with the data source that currently owns the valid tensor values.

        If an Adam state tensor such as ``exp_avg`` or ``exp_avg_sq`` has been
        offloaded, the live device tensor may only be a placeholder with its
        storage released.  In that case, write a cloned CPU mirror into the
        exported state dict so checkpoints contain the real optimizer values.
        Non-swappable state, metadata, and tensors that are still resident on
        device are deep-copied from the original Torch state dict unchanged.
        """
        exported = {
            key: copy.deepcopy(value)
            for key, value in state_dict.items()
            if key not in ("state", "param_groups")
        }
        exported["param_groups"] = copy.deepcopy(state_dict.get("param_groups", []))
        exported["state"] = {}
        saved_groups = exported.get("param_groups", [])
        params_in_order = []
        for group in self.optimizer.param_groups:
            params_in_order.extend(group["params"])
        ids_in_order = []
        for group in saved_groups:
            ids_in_order.extend(group["params"])
        for param, param_id in zip(params_in_order, ids_in_order):
            saved_state = state_dict.get("state", {}).get(param_id)
            if not saved_state:
                continue
            exported_state = {}
            for key in self._state_keys_for_param(param):
                slot = self._slots.get((id(param), key))
                if self.has_host_mirror(slot, key, saved_state):
                    exported_state[key] = slot.cpu_tensor.detach().clone()
            for key, value in saved_state.items():
                if key not in exported_state:
                    exported_state[key] = copy.deepcopy(value)
            exported["state"][param_id] = exported_state
        return exported

    @staticmethod
    def has_host_mirror(slot: Optional[SwapSlot], key: str, saved_state: Dict[str, Any]) -> bool:
        """Return whether ``slot`` currently owns the authoritative CPU mirror for ``key``.

        A slot qualifies only when it has been offloaded to host *and* still
        holds a CPU mirror: after :meth:`export_swappable_state`'s device-side
        fast path, the live tensor may be a storage-released placeholder, so the
        CPU mirror is the only source of the real optimizer values.
        """
        return (
            slot is not None
            and slot.state == "host"
            and slot.cpu_tensor is not None
            and key in saved_state
        )

    def strip_swappable_state(self, state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
        """Split Adam state tensors out before delegating to Torch loading.

        PyTorch's ``load_state_dict`` eagerly restores tensors into the
        optimizer state.  For swap-managed Adam buffers, that would bypass the
        adapter/runtime bookkeeping and can place large tensors directly on the
        device.  This method therefore copies the checkpoint state dict's
        *containers*, removes the Adam buffers that may be swap-managed, and
        returns them in a side table keyed by the checkpoint parameter id.

        Only the container skeleton is rebuilt: leaf tensors are shared with the
        caller's checkpoint instead of being duplicated.  The removed buffers are
        consumed by :meth:`load_swappable_state`, which either takes a reference
        to the checkpoint storage (CPU checkpoints, where ``to(device="cpu")`` is
        a no-op) or copies it once into a host mirror.  Deep-copying the whole
        state dict would hold one extra full copy of every optimizer tensor on
        host memory for the duration of the load, which for large models is a
        needless doubling of the checkpoint-loading peak.

        The stripped state dict is safe to pass to the wrapped optimizer's
        ``load_state_dict`` for ordinary fields such as parameter groups and
        step counters: Torch does not mutate the tensors it reads from it.  The
        removed tensors must be handed to ``load_swappable_state`` afterwards so
        they can be restored with the correct CPU mirror/device placeholder
        layout.
        """
        stripped = self._copy_state_dict_containers(state_dict)
        removed: Dict[int, Dict[str, Any]] = {}
        swappable_keys = self._configured_state_keys()
        for param_id, saved_state in list(stripped.get("state", {}).items()):
            if not isinstance(saved_state, dict):
                continue
            for key in swappable_keys:
                if key in saved_state:
                    removed.setdefault(param_id, {})[key] = saved_state.pop(key)
        return stripped, removed

    @staticmethod
    def _copy_state_dict_containers(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Shallow-copy a checkpoint state dict, sharing every leaf tensor.

        Rebuilds only the nesting Torch's ``load_state_dict`` walks: the top
        level, the ``state`` per-parameter mappings, and the ``param_groups``
        entries.  Do not add tensor copies here: this helper exists to keep the
        checkpoint-loading host peak free of a duplicate optimizer state.
        """
        state = state_dict.get("state")
        copied_state = (
            {param_id: dict(saved_state) if isinstance(saved_state, dict) else saved_state
             for param_id, saved_state in state.items()}
            if isinstance(state, dict)
            else state
        )
        param_groups = state_dict.get("param_groups")
        copied_groups = (
            [dict(group) if isinstance(group, dict) else group for group in param_groups]
            if isinstance(param_groups, list)
            else param_groups
        )
        copied = {key: value for key, value in state_dict.items() if key not in ("state", "param_groups")}
        if state is not None:
            copied["state"] = copied_state
        if param_groups is not None:
            copied["param_groups"] = copied_groups
        return copied

    def load_swappable_state(self, original_state_dict: Dict[str, Any], removed: Dict[int, Dict[str, Any]]) -> None:
        """Restore removed Adam buffers under swap runtime control.

        This is the second half of checkpoint loading.  After Torch has loaded
        the stripped state dict, this method maps checkpoint parameter ids back
        to the current parameter objects by walking saved and current parameter
        groups in order.  Each removed Adam buffer is then recreated as an
        optimizer state entry and registered as a ``SwapSlot``.

        Packed runtimes place checkpoint values directly in persistent pinned
        host views. Legacy runtimes retain an empty device placeholder whose
        storage is restored only during prefetch. Buffers that do not meet the
        runtime's swappability criteria are materialized directly on the
        parameter's device and tracked as normal device-resident slots.
        """
        saved_groups = original_state_dict.get("param_groups", [])
        self._slots = {}
        # Walk saved and current parameter groups in order to map checkpoint ids
        # back to the live parameter objects.
        saved_ids = [
            saved_id
            for saved_group, current_group in zip(saved_groups, self.optimizer.param_groups)
            for saved_id in saved_group["params"]
        ]
        groups = self.optimizer.param_groups
        current_params = [param for group in groups for param in group["params"]]
        for saved_id, param in zip(saved_ids, current_params):
            for key, saved_tensor in removed.get(saved_id, {}).items():
                self._restore_swappable_entry(param, saved_id, saved_groups, key, saved_tensor)
        if self.runtime.packed_enabled:
            self.runtime.prepare_packed_host(self._ordered_slots())
            self.publish_packed_state()

    def _restore_swappable_entry(
            self,
            param: Any,
            saved_id: int,
            saved_groups: List[Dict[str, Any]],
            key: str,
            saved_tensor: Any,
    ) -> None:
        """Restore one removed checkpoint buffer as a host or device-resident slot."""
        state = self.optimizer.state[param]
        cpu_tensor = self._cast_swappable_tensor_to_cpu(param, saved_tensor)
        if self.runtime.packed_enabled and self.runtime.is_packable_template(param, self.config.min_numel):
            if self.runtime.is_distributed_tensor(param):
                logical_tensor = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
                slot = self._make_slot(key, logical_tensor)
                state[key] = logical_tensor
                self.runtime.release_device_storage(slot)
            else:
                slot = self._make_slot(key, None, template=param)
                state[key] = cpu_tensor
                slot.tensor = cpu_tensor
            slot.cpu_tensor = cpu_tensor
            slot.state = "host"
            self._slots[(id(param), key)] = slot
            return
        device_tensor = self.runtime.make_empty_device_tensor_like(param, cpu_tensor)
        slot = self._make_slot(key, device_tensor)
        if slot.swappable:
            state[key] = device_tensor
            slot.cpu_tensor = self.runtime.make_cpu_tensor(cpu_tensor)
            slot.state = "host"
            self._slots[(id(param), key)] = slot
            self.runtime.release_device_storage(slot)
        else:
            device_tensor = self._cast_state_tensor_like_torch(
                param,
                saved_tensor,
                saved_id,
                saved_groups,
                key,
            )
            state[key] = device_tensor
            self._slots[(id(param), key)] = self._make_slot(key, device_tensor)

    @staticmethod
    def _cast_state_tensor_like_torch(
            param: Any,
            saved_tensor: Any,
            saved_id: int,
            saved_groups: List[Dict[str, Any]],
            key: str,
    ) -> Any:
        """Cast a loaded state tensor using PyTorch optimizer load semantics."""
        if not isinstance(saved_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in optimizer state, got {type(saved_tensor)!r}.")
        process = getattr(torch.optim.Optimizer, "_process_value_according_to_param_policy", None)
        if process is not None:
            return process(param, saved_tensor, saved_id, saved_groups, key).detach().clone()
        if key == "step":
            return saved_tensor.detach().clone()
        if param.is_floating_point():
            return saved_tensor.detach().to(dtype=param.dtype, device=param.device).clone()
        return saved_tensor.detach().to(device=param.device).clone()

    @staticmethod
    def _cast_swappable_tensor_to_cpu(param: Any, saved_tensor: Any) -> Any:
        """Cast swappable state dtype like PyTorch while keeping values on CPU."""
        if not isinstance(saved_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in optimizer state, got {type(saved_tensor)!r}.")
        if param.is_floating_point():
            return saved_tensor.detach().to(dtype=param.dtype, device="cpu")
        return saved_tensor.detach().to(device="cpu")

    def _register_present_slots(self, param: Any, state: Dict[str, Any]) -> None:
        """Register configured state tensors that already exist in an optimizer state mapping."""
        for key in self._configured_state_keys():
            tensor = state.get(key)
            if tensor is None or (id(param), key) in self._slots:
                continue
            self._slots[(id(param), key)] = self._make_slot(key, tensor)

    def _build_slots(self, param: Any, state: Dict[str, Any]) -> List[SwapSlot]:
        """Return swap slots associated with the current parameter state."""
        slots = []
        for key in self._state_keys_for_param(param):
            tensor = state.get(key)
            if tensor is None:
                continue
            slot = self._slots.get((id(param), key))
            if slot is None:
                slot = self._make_slot(key, tensor)
                self._slots[(id(param), key)] = slot
            elif slot.tensor is not tensor and slot.cpu_tensor is not tensor:
                slot = self._make_slot(key, tensor)
                self._slots[(id(param), key)] = slot
            slots.append(slot)
        return slots

    def _make_slot(self, key: str, tensor: Any, template: Optional[Any] = None) -> SwapSlot:
        """Create a swap slot for a state tensor or a packed-state template."""
        metadata_tensor = tensor if tensor is not None else template
        if metadata_tensor is None:
            raise ValueError(f"Cannot build swap slot {key!r} without a tensor or template.")
        if tensor is None:
            swappable = self.runtime.is_packable_template(metadata_tensor, self.config.min_numel)
        else:
            swappable = self.runtime.is_swappable_tensor(tensor, self.config.min_numel)
        packed = bool(self.runtime.packed_enabled and swappable)
        slot = SwapSlot(
            name=key,
            tensor=tensor,
            cpu_tensor=None,
            swappable=swappable,
            state="device" if tensor is not None else "pending",
            packed=packed,
            logical_tensor=tensor if packed and self.runtime.is_distributed_tensor(tensor) else None,
        )
        self.runtime.populate_slot_metadata(slot, metadata_tensor)
        return slot

    def _ordered_slots(self) -> List[SwapSlot]:
        """Return slots in optimizer parameter and configured state-key order."""
        slots = []
        seen_slots = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                for key in self._configured_state_keys():
                    slot = self._slots.get((id(param), key))
                    if slot is not None and id(slot) not in seen_slots:
                        slots.append(slot)
                        seen_slots.add(id(slot))
        return slots

    def _state_keys_for_param(self, param: Any) -> Tuple[str, ...]:
        state = self.optimizer.state[param]
        keys = self._configured_state_keys()
        result = []
        for key in keys:
            if key in state:
                result.append(key)
            elif self.config.state_keys is not None:
                raise ValueError(f"Requested state key '{key}' is not present for parameter.")
        return tuple(result)

    def _configured_state_keys(self) -> Tuple[str, ...]:
        """Return Adam state keys selected for swap by the current config."""
        keys = self.config.state_keys or self.default_state_keys()
        result = []
        for key in keys:
            if key == "master_param":
                if self.config.state_keys is not None:
                    raise ValueError(f"Requested state key '{key}' is not available for {type(self.optimizer)!r}.")
                continue
            result.append(key)
        return tuple(result)


class SwapOptimizer(torch.optim.Optimizer):
    """Torch optimizer wrapper around a family-provided swap adapter.

    The wrapper is algorithm agnostic: it owns the pipeline runtime, exposes the
    wrapped optimizer's ``param_groups``/``state``/``defaults`` and drives one
    step as ``prepare_step`` -> ``partition`` -> ``run_pipeline`` ->
    ``finish_step``.  The concrete adapter is built by the family factory passed
    as ``adapter_factory`` (see ``swap_adam.py``/``swap_muon.py``).
    """

    _is_swap_optimizer = True

    def __init__(self, optimizer: Any, config: Any, adapter_factory: AdapterFactory) -> None:
        # Do not call ``torch.optim.Optimizer.__init__``: the wrapped base
        # optimizer already owns param_groups/state/defaults. Inheriting keeps
        # PyTorch LR schedulers and isinstance checks happy while this wrapper
        # delegates all optimizer state to ``self.optimizer``.
        self.optimizer = optimizer
        self.config = config
        self.runtime = PipelineSwapRuntime(config)
        self.adapter = adapter_factory(optimizer, config, self.runtime)
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

    @contextlib.contextmanager
    def _no_grad_context(self) -> Iterable[None]:
        with torch.no_grad():
            yield
