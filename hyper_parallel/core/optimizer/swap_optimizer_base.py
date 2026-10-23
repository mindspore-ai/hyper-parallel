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
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

# Optimizer runtime is Torch-only; the legacy checker still flags core imports.
import torch  # pylint: disable=forbidden-backend-import

logger = logging.getLogger(__name__)

# Logical optimizer-state keys every swap family in this codebase understands.
# This is the vocabulary ``SwapOptimizerConfig.state_keys`` may name; each
# adapter still validates that a key belongs to its own optimizer family.
KNOWN_STATE_KEYS = (
    "exp_avg",
    "exp_avg_sq",
    "max_exp_avg_sq",
    "momentum_buffer",
)
_PACKED_ALIGNMENT_BYTES = 512
_DEVICE_TYPE = "npu"
_MISSING_CHECKPOINT_STATE = object()


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
        if self.logical_tensor is not None:
            # DTensor exposes no public setter for replacing its local shard.
            # Rebind the shell on host too, so it cannot retain an arena view.
            setattr(self.logical_tensor, "_local_tensor", tensor)
            self.logical_tensor.data = tensor
        if self.logical_tensor is not None and device_type != "cpu":
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
        """Initialize instance-owned streams and packed storage metadata."""
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
        self._packed_slot_bind_hook: Optional[Callable[[SwapSlot], None]] = None

    @property
    def packed_enabled(self) -> bool:
        """Return whether this runtime may build packed state candidates."""
        return self._packed_enabled

    @property
    def copy_stream(self) -> Optional[Any]:
        """Return the transfer stream after its lazy initialization."""
        return self._copy_stream

    def set_packed_slot_bind_hook(self, hook: Callable[[SwapSlot], None]) -> None:
        """Install the adapter hook that republishes a packed slot's live view."""
        self._packed_slot_bind_hook = hook

    def _bind_packed_slot(self, slot: SwapSlot, tensor: Any) -> None:
        """Bind a packed view and notify the owning state adapter."""
        slot.bind_tensor(tensor)
        if self._packed_slot_bind_hook is not None:
            self._packed_slot_bind_hook(slot)

    def partition(
            self,
            units: Sequence[SwapUpdateUnit],
            unit_costs: Optional[Sequence[int]] = None,
    ) -> List[List[SwapUpdateUnit]]:
        """Partition consecutive units using explicit costs or local state bytes.

        Args:
            units: Atomic updates in execution order, including empty local units.
            unit_costs: Optional non-negative integer costs, one per unit. A
                distributed adapter supplies rank-independent planning weights.

        Returns:
            Consecutive batches retaining every unit in its original order.

        Raises:
            ValueError: If explicit costs have the wrong length or invalid values.
        """
        ordered_units = list(units)
        if unit_costs is None:
            unit_costs = [self._unit_cost(unit) for unit in ordered_units]
        else:
            unit_costs = list(unit_costs)
            if len(unit_costs) != len(ordered_units):
                raise ValueError("unit_costs must contain exactly one cost per update unit.")
            if any(isinstance(cost, bool) or not isinstance(cost, int) or cost < 0 for cost in unit_costs):
                raise ValueError("unit_costs must contain non-negative integers.")
        if not ordered_units:
            return []
        swap_times = max(1, min(int(self.config.swap_times), len(ordered_units)))
        unit_costs = [max(1, cost) for cost in unit_costs]
        remaining_cost = sum(unit_costs)
        batches: List[List[SwapUpdateUnit]] = []
        start = 0
        for batch_index in range(swap_times - 1):
            remaining_batches = swap_times - batch_index

            # Compare against the remaining average without introducing floating-point rounding.
            # Bind the loop state as defaults so the closure captures this iteration's values.
            def should_stop_growing(
                    end: int,
                    current_cost: int,
                    next_cost: int,
                    remaining_batches: int = remaining_batches,
                    remaining_cost: int = remaining_cost,
                    start: int = start,
            ) -> bool:
                """Return whether adding the next unit would not improve balance."""
                current_distance = abs(remaining_cost - current_cost * remaining_batches)
                next_distance = abs(remaining_cost - (current_cost + next_cost) * remaining_batches)
                return end > start and current_distance <= next_distance

            max_end = len(ordered_units) - remaining_batches + 1
            end, current_cost = self._grow_balanced_segment(unit_costs, start, max_end, should_stop_growing)
            batches.append(ordered_units[start:end])
            remaining_cost -= current_cost
            start = end
        batches.append(ordered_units[start:])
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
        self.restore_packed_host_views(slot_list)

    def materialize_slots(self, slots: Iterable[SwapSlot]) -> None:
        """Materialize host-resident slots on their original device.

        HSDP checkpoint broadcasts operate on device tensors.  Swap slots are
        normally host-resident between optimizer steps, so checkpoint code uses
        this helper to reuse the normal H2D stream/event ordering before issuing
        a collective.
        """
        slot_list = [
            slot for slot in _iter_unique_slot_objects(slots)
            if slot.swappable and slot.state == "host"
        ]
        if not slot_list:
            return
        for slot in slot_list:
            if slot.packed:
                device_tensor = torch.empty(slot.shape, dtype=slot.dtype, device=slot.device)
                self._bind_packed_slot(slot, device_tensor)
        self.prefetch([UpdateUnit(0, None, None, slot_list)])
        self.wait_prefetch([UpdateUnit(0, None, None, slot_list)])

    def restore_packed_host_views(self, slots: Iterable[SwapSlot]) -> None:
        """Restore persistent packed host views after temporary materialization."""
        for slot in _iter_unique_slot_objects(slots):
            if (slot.packed and slot.state == "host" and slot.cpu_tensor is not None
                    and slot.tensor is not slot.cpu_tensor):
                self._bind_packed_slot(slot, slot.cpu_tensor)

    def prefetch(self, batch: Sequence[UpdateUnit]) -> None:
        """Prefetch batch slots from CPU to device."""
        slots = [slot for slot in _iter_unique_slots(batch) if slot.swappable and slot.state == "host"]
        if not slots:
            return

        for slot in slots:
            if slot.cpu_tensor is None:
                raise RuntimeError(f"Swap slot {slot.name!r} is host-resident but has no CPU mirror.")

        restored_slots = []
        try:
            for slot in slots:
                self.restore_device_storage(slot)
                restored_slots.append(slot)
        except Exception:
            for restored_slot in reversed(restored_slots):
                self.release_device_storage(restored_slot)
            raise

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

    def prepare_packed_host(self, slots: Sequence[SwapSlot]) -> None:
        """Pack logical optimizer states into persistent pinned buffers by dtype."""
        if not self._packed_enabled:
            return

        packed_slots = [slot for slot in slots if slot.swappable and slot.packed]
        signature = self._validate_packed_host_layout(packed_slots)
        if signature == self._host_layout_signature:
            return

        self._prepare_packed_host_sources(packed_slots)
        slots_by_dtype: Dict[Any, List[SwapSlot]] = {}
        for slot in packed_slots:
            slots_by_dtype.setdefault(slot.dtype, []).append(slot)

        self._host_buffers = {
            dtype: self._pack_dtype_slots(dtype, dtype_slots)
            for dtype, dtype_slots in slots_by_dtype.items()
        }
        self._host_layout_signature = signature

    def _validate_packed_host_layout(
            self, packed_slots: Sequence[SwapSlot],
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Validate candidates and return their persistent layout signature."""
        if self._packed_batch_plans or self._packed_tail_event is not None:
            raise RuntimeError("Cannot rebuild packed optimizer layout during an active transfer.")
        slot_ids = [id(slot) for slot in packed_slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise RuntimeError("Packed optimizer layout contains a duplicate state slot.")
        return tuple(
            (id(slot), slot.dtype, slot.shape, slot.numel, slot.device)
            for slot in packed_slots
        )

    def _prepare_packed_host_sources(self, packed_slots: Sequence[SwapSlot]) -> None:
        """Validate source layouts and settle copies before rebuilding host buffers."""
        for slot in packed_slots:
            source = slot.checkpoint_tensor
            if source is not None:
                local = self._storage_tensor(source)
                if local.layout != torch.strided or not local.is_contiguous():
                    raise ValueError("Packed optimizer states must be dense and contiguous before host packing.")
        # A layout rebuild reads CPU mirrors on the host. Stream ordering alone
        # is insufficient, including after an initial-state offload.
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        for event in _iter_unique_events(packed_slots):
            self.wait_event(event, None)
        if any(slot.state in ("h2d", "d2h") for slot in packed_slots):
            self.synchronize_cpu_mirrors(packed_slots)

    def invalidate_packed_layout(self) -> None:
        """Discard an idle packed layout after state slots have been rebound.

        Persistent slot mirrors retain their values until the next host packing.
        The instance's copy stream remains available for subsequent steps.
        """
        if self._packed_batch_plans or self._packed_tail_event is not None:
            raise RuntimeError("Cannot reset optimizer parameters during a packed transfer.")
        for arena in self._staging_arenas:
            if arena is not None:
                arena.dtype_views.clear()
                arena.raw_buffer.untyped_storage().resize_(0)
        self._staging_arenas = [None, None]
        self._packed_device_views.clear()
        self._packed_ready_events.clear()
        self._packed_offload_events.clear()
        self._host_buffers = {}
        self._host_layout_signature = ()

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
            source = slot.tensor if slot.state == "device" else slot.checkpoint_tensor
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

    def make_empty_state_like(self, param: Any) -> Any:
        """Allocate a state shell with the parameter's local and logical metadata."""
        local = torch.empty_like(self._storage_tensor(param), memory_format=torch.preserve_format)
        if not self.is_distributed_tensor(param):
            return local
        layout = getattr(param, "layout", None)
        if hasattr(layout, "tensor_stride"):
            logical_stride = layout.tensor_stride
        else:
            logical_stride = tuple(param.stride())
        if logical_stride is None:
            logical_stride = [1] * len(param.shape)
            for index in range(len(logical_stride) - 2, -1, -1):
                logical_stride[index] = logical_stride[index + 1] * max(int(param.shape[index + 1]), 1)
        # Explicit from_local preserves layout even inside SkipDTensorDispatch,
        # where empty_like(param) would otherwise return a plain local tensor.
        return type(param).from_local(
            local, device_mesh=param.device_mesh, placements=param.placements,
            run_check=False, shape=param.shape, stride=tuple(logical_stride),
        )

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

    def stream_context(self, stream: Any) -> Any:
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
        slots = itertools.chain.from_iterable(unit.slots for unit in units)
        swappable_slots = [slot for slot in slots if slot.swappable]
        if not swappable_slots:
            return False
        if len({id(slot) for slot in swappable_slots}) != len(swappable_slots):
            raise RuntimeError("Packed optimizer batches contain a duplicate state slot.")
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
        slots = itertools.chain.from_iterable(unit.slots for unit in units)
        swappable = (slot for slot in slots if slot.swappable)
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
                self._bind_packed_slot(slot, device_view)
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
            # This arena may be overwritten by the following H2D. Exception
            # cleanup must use the host result once its D2H has been submitted.
            for region in self._packed_batch_plans[batch_index].regions.values():
                for slot in region.slots:
                    slot.state = "d2h"
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
                self._bind_packed_slot(slot, slot.cpu_tensor)
                slot.state = "host"
                slot.event = None

    def end_packed_step(self) -> None:
        """Drain the packed copy chain and release step-local staging storage."""
        if self._copy_stream is not None:
            # A submission can fail after enqueueing a copy but before recording
            # its tail event, even when no batch has activated a device view yet.
            self._copy_stream.synchronize()
        elif self._packed_tail_event is not None:
            self.wait_event(self._packed_tail_event, None)
        active_slots: Dict[int, Any] = {}
        for batch_plan in self._packed_batch_plans:
            for region in batch_plan.regions.values():
                for slot in region.slots:
                    if slot.state in ("device", "d2h"):
                        active_slots[id(slot)] = slot
        if active_slots:
            compute_stream = self.current_stream()
            synchronize = getattr(compute_stream, "synchronize", None)
            if synchronize is not None:
                synchronize()
            for slot in active_slots.values():
                if slot.state == "device":
                    slot.cpu_tensor.copy_(self._storage_tensor(slot.tensor).detach(), non_blocking=False)
                self._bind_packed_slot(slot, slot.cpu_tensor)
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
                if not slot.swappable or not slot.packed:
                    continue
                if id(slot) in seen_slots:
                    raise RuntimeError("Packed optimizer batch contains a duplicate state slot.")
                slots_by_dtype.setdefault(slot.dtype, []).append(slot)
                seen_slots.add(id(slot))

        regions = {}
        for dtype, slots in slots_by_dtype.items():
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

    @staticmethod
    def default_state_keys() -> Tuple[str, ...]:
        """Return the family's default swap state keys.

        Subclasses must override this; the base class deliberately assumes
        nothing about an optimizer's state layout, so there is no default.

        Raises:
            NotImplementedError: Always, unless a subclass overrides it.
        """
        raise NotImplementedError

    def __init__(self, optimizer: Any, config: Any, runtime: Any) -> None:
        """Bind a leaf optimizer to its state slots and transfer runtime."""
        self.optimizer = optimizer
        self.config = config
        self.runtime = runtime
        self._slots: Dict[Tuple[int, str], SwapSlot] = {}
        self._packed_state_bindings: Dict[int, Tuple[Any, str]] = {}
        self._checkpoint_active = False
        self._checkpoint_broadcasted = False
        self._checkpoint_state_bindings: List[Tuple[Any, str, Any]] = []
        self.runtime.set_packed_slot_bind_hook(self.on_packed_slot_bound)

    def iter_update_units(self, step_context: Dict[str, Any]) -> List[UpdateUnit]:
        """Return units collected in ``prepare_step``."""
        return step_context["units"]

    def partition_costs(
            self,
            units: Sequence[SwapUpdateUnit],
            step_context: Any,
    ) -> Optional[Sequence[int]]:
        """Return optional planning costs; default to the runtime's local bytes."""
        del units, step_context

    def prepare_pipeline(
            self,
            batches: Sequence[Sequence[SwapUpdateUnit]],
            step_context: Any,
    ) -> None:
        """Pack local slots in execution order after logical partitioning."""
        del step_context
        if not self.runtime.packed_enabled:
            return
        self._refresh_packed_state_bindings()
        active_slots = [slot for batch in batches for unit in batch for slot in unit.slots]
        active_ids = {id(slot) for slot in active_slots}
        # Inactive state stays on host at the layout tail. It must not leave
        # holes inside active dtype regions or keep old packed buffers alive.
        inactive_slots = [slot for slot in self._ordered_slots() if id(slot) not in active_ids]
        self.runtime.prepare_packed_host(active_slots + inactive_slots)
        self.publish_packed_state()

    def _refresh_packed_state_bindings(self) -> None:
        """Index packed slots by their live optimizer-state owner."""
        bindings = {}
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                for key in self._configured_state_keys():
                    slot = self._slots.get((id(param), key))
                    if slot is not None and slot.packed:
                        bindings[id(slot)] = (param, key)
        self._packed_state_bindings = bindings

    def on_packed_slot_bound(self, slot: SwapSlot) -> None:
        """Publish a packed host or staging view into optimizer state."""
        binding = self._packed_state_bindings.get(id(slot))
        if binding is None:
            raise RuntimeError(f"Packed optimizer slot {slot.name!r} has no live state binding.")
        param, key = binding
        state = self.optimizer.state.get(param)
        if state is None:
            raise RuntimeError(f"Packed optimizer slot {slot.name!r} has no optimizer state entry.")
        state[key] = slot.tensor

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

    def _checkpoint_broadcast_state_keys(self) -> Tuple[str, ...]:
        """Return the wrapped optimizer state fields consumed by its broadcast."""
        get_state_keys = getattr(self.optimizer, "_state_keys_for_checkpoint_broadcast", None)
        if not callable(get_state_keys):
            return ()
        return tuple(get_state_keys())

    def _make_checkpoint_cpu_tensor(self, param: Any) -> torch.Tensor:
        """Allocate one host receiver matching the optimizer broadcast contract."""
        local_param = self.runtime._storage_tensor(param)
        try:
            return torch.empty_like(local_param, dtype=torch.float32, device="cpu", pin_memory=True)
        except RuntimeError:
            return torch.empty_like(local_param, dtype=torch.float32, device="cpu")

    def _publish_checkpoint_cpu_state(self) -> None:
        """Expose CPU mirrors and host receivers to the fused state broadcast.

        The wrapped broadcast packs these tensors into one bounded device buffer
        per batch. Existing swap slots therefore stay host-resident, while ranks
        without owner state receive directly into temporary CPU tensors.
        """
        broadcast_info = getattr(self.optimizer, "_param_to_broadcast_info", None)
        state_keys = self._checkpoint_broadcast_state_keys()
        if not broadcast_info or not state_keys:
            return

        for param in broadcast_info:
            param_state = self.optimizer.state.get(param)
            if param_state is None:
                param_state = {}
            for key in state_keys:
                slot = self._slots.get((id(param), key))
                replacement = None
                if slot is not None and slot.swappable:
                    if slot.state != "host" or slot.cpu_tensor is None:
                        raise RuntimeError(
                            f"Checkpoint state slot {key!r} must have an authoritative CPU mirror."
                        )
                    replacement = slot.cpu_tensor
                elif key not in param_state:
                    replacement = self._make_checkpoint_cpu_tensor(param)

                if replacement is None:
                    continue
                if param not in self.optimizer.state:
                    self.optimizer.state[param] = param_state
                previous = param_state.get(key, _MISSING_CHECKPOINT_STATE)
                self._checkpoint_state_bindings.append((param, key, previous))
                param_state[key] = replacement

    def _restore_checkpoint_state_bindings(self) -> None:
        """Restore live state entries after exporting the CPU checkpoint values."""
        for param, key, previous in reversed(self._checkpoint_state_bindings):
            param_state = self.optimizer.state.get(param)
            if param_state is None:
                continue
            if previous is _MISSING_CHECKPOINT_STATE:
                param_state.pop(key, None)
                if not param_state:
                    self.optimizer.state.pop(param, None)
            else:
                param_state[key] = previous
        self._checkpoint_state_bindings.clear()

    def prepare_checkpoint(self) -> None:
        """Publish host-resident swap state for an optional HSDP broadcast."""
        if self._checkpoint_active:
            return
        self._checkpoint_active = True
        self._checkpoint_broadcasted = False
        try:
            self._refresh_packed_state_bindings()
            # A previous optimizer step may still have a completed D2H event or
            # a live device state. Make the host mirror authoritative first;
            # the broadcast streams it through its bounded device buffer.
            self.runtime.synchronize_cpu_mirrors(self.all_slots())
            self._publish_checkpoint_cpu_state()
        except Exception:
            try:
                self.runtime.synchronize_cpu_mirrors(self.all_slots())
            except RuntimeError:  # pragma: no cover - best effort after a transfer failure
                logger.exception("Failed to restore swap slots after checkpoint preparation failed")
            finally:
                self._restore_checkpoint_state_bindings()
                self._checkpoint_active = False
                self._checkpoint_broadcasted = False
            raise

    def finish_checkpoint(self) -> None:
        """Return checkpoint-broadcast state to the normal host-resident form."""
        if not self._checkpoint_active:
            return
        try:
            self.runtime.synchronize_cpu_mirrors(self.all_slots())
        finally:
            self._restore_checkpoint_state_bindings()
            self._checkpoint_active = False
            self._checkpoint_broadcasted = False

    def all_slots(self) -> Iterable[SwapSlot]:
        """Iterate known swap slots."""
        return tuple(self._ordered_slots())

    def tracked_slots(self) -> Iterable[SwapSlot]:
        """Return all registered slots, including parameters removed from groups."""
        return tuple({id(slot): slot for slot in self._slots.values()}.values())

    def rebind_parameter_slots(self, new_to_old_param: Dict[Any, Any]) -> None:
        """Rebuild state bindings using an explicit parameter identity mapping.

        Args:
            new_to_old_param: New-to-old identities; unrelated leaf entries are
                ignored. CPU mirrors must already be synchronized.

        Raises:
            ValueError: If a state cannot be mapped to the new local shape.
        """
        params = [param for group in self.optimizer.param_groups for param in group["params"]]
        local_ids = {id(param) for param in self._params_with_local_state(params)}
        rebound_slots = {}
        rebound_state = {}
        for param in params:
            if id(param) not in local_ids:
                continue
            old_param = new_to_old_param.get(param, param)
            state = self.optimizer.state.get(param, self.optimizer.state.get(old_param))
            if state is None:
                continue
            state = dict(state)
            for key in self._configured_state_keys():
                if key not in state:
                    continue
                slot = self._slots.get((id(param), key))
                if slot is None:
                    slot = self._slots.get((id(old_param), key))
                slot = self._rebind_state_slot(param, key, state[key], slot, param is not old_param)
                state[key] = slot.tensor
                rebound_slots[(id(param), key)] = slot
            rebound_state[param] = state

        # Publish only after all sources have been read. A failed conversion must
        # leave the old slot map (and its authoritative CPU mirrors) recoverable.
        for param in list(self.optimizer.state):
            if isinstance(param, torch.Tensor) and param not in rebound_state:
                self.optimizer.state.pop(param)
        self.optimizer.state.update(rebound_state)
        self._slots = rebound_slots
        self._refresh_packed_state_bindings()

    def _rebind_state_slot(
            self, param: Any, key: str, tensor: Any, slot: Optional[SwapSlot], replaced: bool,
    ) -> SwapSlot:
        """Preserve a compatible binding or recreate it from authoritative values."""
        local_param = self.runtime._storage_tensor(param)
        if slot is not None and (tensor is slot.tensor or tensor is slot.cpu_tensor):
            if (not replaced and slot.shape == tuple(local_param.shape)
                    and slot.dtype == local_param.dtype and slot.device == local_param.device
                    and not self.runtime.is_distributed_tensor(param)):
                return slot
            source = slot.checkpoint_tensor
        else:
            # Identity verifies a missing map; it never infers a replacement.
            if any(tensor is tracked.tensor or tensor is tracked.cpu_tensor for tracked in self._slots.values()):
                raise ValueError("Replacing a swap parameter requires an explicit new_to_old_param mapping.")
            source = tensor
        source = self.runtime._storage_tensor(source)
        if source.numel() != local_param.numel():
            raise ValueError(
                f"Cannot rebind {key!r} with local shape {tuple(source.shape)} "
                f"to {tuple(local_param.shape)}; restore state for the new layout first."
            )
        cpu_tensor = self._cast_swappable_tensor_to_cpu(param, source).reshape(local_param.shape)
        device_tensor = self.runtime.make_empty_state_like(param)
        rebound = self._make_slot(key, device_tensor)
        if rebound.swappable:
            rebound.cpu_tensor = self.runtime.make_cpu_tensor(cpu_tensor)
            rebound.state = "host"
            self.runtime.release_device_storage(rebound)
            if rebound.packed:
                rebound.bind_tensor(rebound.cpu_tensor)
        else:
            self.runtime._storage_tensor(device_tensor).copy_(cpu_tensor)
        return rebound

    def checkpoint_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return optimizer checkpoint state using CPU mirrors for swapped slots."""
        del args, kwargs
        self._refresh_packed_state_bindings()
        self.runtime.synchronize_cpu_mirrors(self.all_slots())
        return self.export_swappable_state(self.optimizer.state_dict())

    @contextlib.contextmanager
    def checkpoint_load_context(self, state_dict: Dict[str, Any], flatten: bool) -> Iterable[None]:
        """Provide state keys for FQN loading without an initialization step.

        Args:
            state_dict: Chain checkpoint keyed by model FQN.
            flatten: Whether the checkpoint uses flattened FQN entries.
        """
        previous_state = self.optimizer.state
        schema = copy.copy(previous_state)
        schema.clear()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                name = param.model_name
                if flatten:
                    prefix = f"state.{name}."
                    schema[param] = {
                        key[len(prefix):].split(".", 1)[0]: None
                        for key in state_dict if key.startswith(prefix)
                    }
                else:
                    schema[param] = dict(state_dict.get("state", {}).get(name, {}))
        # DCP uses live state keys to unflatten and calls step() if state is
        # empty. Empty owner ranks must never initiate that extra collective.
        self.optimizer.state = schema
        try:
            yield
        finally:
            if self.optimizer.state is schema:
                self.optimizer.state = previous_state

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

    def _params_with_local_state(self, params: Sequence[Any]) -> Iterable[Any]:
        """Return parameters whose swapped state belongs to this rank.

        The default restores every checkpoint entry. Distributed optimizer
        adapters may filter the sequence to preserve rank-local state ownership.
        """
        return params

    def publish_packed_state(self) -> None:
        """Publish persistent packed CPU mirrors to the wrapped optimizer state."""
        if not self.runtime.packed_enabled:
            return
        self._refresh_packed_state_bindings()
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
            key: self._copy_checkpoint_value(value)
            for key, value in state_dict.items()
            if key not in ("state", "param_groups")
        }
        exported["param_groups"] = self._copy_checkpoint_value(
            state_dict.get("param_groups", [])
        )
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
                    exported_state[key] = self._copy_checkpoint_value(value)
            exported["state"][param_id] = exported_state
        return exported

    @staticmethod
    def _copy_checkpoint_value(value: Any) -> Any:
        """Clone checkpoint values without dispatching ``deepcopy`` on DTensor.

        HSDP state broadcasts may leave a distributed tensor in the wrapped
        optimizer state.  Hyper DTensor intentionally rejects unregistered
        ``__deepcopy__`` dispatch, so normalize distributed tensors to their
        local shards before cloning.  Nested containers are copied recursively;
        scalar metadata still uses regular ``copy.deepcopy`` semantics.
        """
        if isinstance(value, torch.Tensor):
            to_local = getattr(value, "to_local", None)
            local = to_local() if callable(to_local) else value
            if local is not value:
                return local.detach().cpu().clone()
            return copy.deepcopy(value)
        if isinstance(value, dict):
            return {
                key: StateSwapAdapter._copy_checkpoint_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [StateSwapAdapter._copy_checkpoint_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(StateSwapAdapter._copy_checkpoint_value(item) for item in value)
        return copy.deepcopy(value)

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
        local_param_ids = {
            id(param) for param in self._params_with_local_state(current_params)
        }
        for saved_id, param in zip(saved_ids, current_params):
            if id(param) not in local_param_ids:
                self.optimizer.state.pop(param, None)
                continue
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
                logical_tensor = self.runtime.make_empty_state_like(param)
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
        """Return optimizer state keys selected for swap by the current config."""
        return tuple(self.config.state_keys or self.default_state_keys())


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
        """Wrap a leaf with its family adapter and independent transfer runtime."""
        # Do not call ``torch.optim.Optimizer.__init__``: the wrapped base
        # optimizer already owns param_groups/state/defaults. Inheriting keeps
        # PyTorch LR schedulers and isinstance checks happy while this wrapper
        # delegates all optimizer state to ``self.optimizer``.
        self.optimizer = optimizer
        self.config = config
        self._step_active = False
        self._reset_active = False
        self._reset_required = False
        self._export_active = False
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
    def param_groups(self) -> List[Dict[str, Any]]:
        """Proxy parameter groups."""
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: List[Dict[str, Any]]) -> None:
        """Replace the underlying optimizer's parameter groups."""
        self.optimizer.param_groups = value

    @property
    def state(self) -> Dict[Any, Any]:
        """Proxy optimizer state."""
        return self.optimizer.state

    @property
    def defaults(self) -> Dict[str, Any]:
        """Proxy optimizer defaults."""
        return self.optimizer.defaults

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Proxy param group addition."""
        self.optimizer.add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Proxy gradient clearing."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def reset_optimizer_parameters(
            self, new_to_old_param: Optional[Dict[torch.nn.Parameter, torch.nn.Parameter]] = None,
    ) -> None:
        """Rebuild optimizer metadata and rebind state at an idle step boundary.

        Args:
            new_to_old_param: Explicit new-to-old identities after param_groups
                replacement. Omit when parameter identities are unchanged.

        Raises:
            RuntimeError: If a step, checkpoint or transfer is still active.
        """
        slots = tuple(self.adapter.tracked_slots())
        if (self._step_active or self._reset_active or self._export_active or self.adapter._checkpoint_active
                or self.runtime._packed_batch_plans or self.runtime._packed_tail_event is not None
                or any(slot.state in ("h2d", "d2h") for slot in slots)):
            raise RuntimeError("Cannot reset optimizer parameters during a step, checkpoint or transfer.")
        self._reset_active = True
        self._reset_required = True
        try:
            with self._no_grad_context():
                self.runtime.synchronize_cpu_mirrors(slots)
                reset = getattr(self.optimizer, "reset_optimizer_parameters", None)
                if callable(reset):
                    reset()
                self.adapter.rebind_parameter_slots(new_to_old_param or {})
                self.runtime.invalidate_packed_layout()
            self._reset_required = False
        finally:
            self._reset_active = False

    def step(self, closure: Optional[Any] = None) -> Any:
        """Run one optimizer step with pipeline state swap."""
        if closure is not None:
            raise ValueError("Swap optimizer does not support closure.")
        self._check_reset_complete()
        self._step_active = True
        try:
            with self._no_grad_context():
                step_context = self.adapter.prepare_step()
                units = self.adapter.iter_update_units(step_context)
                unit_costs = self.adapter.partition_costs(units, step_context)
                batches = self.runtime.partition(units, unit_costs=unit_costs)
                self.adapter.prepare_pipeline(batches, step_context)
                self.runtime.run_pipeline(batches, step_context, self.adapter.step_batch)
                return self.adapter.finish_step(step_context)
        finally:
            self._step_active = False

    def _check_reset_complete(self) -> None:
        """Prevent using a partially rebuilt optimizer after reset failed."""
        if self._reset_required:
            raise RuntimeError("Optimizer parameter reset has not completed; retry reset before step or checkpoint.")

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dict using CPU mirrors for swappable tensors."""
        self._check_reset_complete()
        self._export_active = True
        try:
            return self._export_state_dict()
        finally:
            self._export_active = False

    def _export_state_dict(self) -> Dict[str, Any]:
        """Reuse an outer chain checkpoint window or open a leaf-owned window."""
        broadcast = getattr(self.optimizer, "_broadcast_state_fused_for_ckpt", None)
        if not callable(broadcast):
            return self.adapter.checkpoint_state_dict()

        owns_checkpoint_window = not self.adapter._checkpoint_active
        try:
            if owns_checkpoint_window:
                self._prepare_swap_checkpoint()
            if not self.adapter._checkpoint_broadcasted:
                self._broadcast_state_fused_for_ckpt()
            return self.adapter.checkpoint_state_dict()
        finally:
            if owns_checkpoint_window:
                self.cleanup_synced_state()

    def _prepare_swap_checkpoint(self) -> None:
        """Publish CPU swap state needed by an HSDP checkpoint broadcast."""
        self._check_reset_complete()
        broadcast = getattr(self.optimizer, "_broadcast_state_fused_for_ckpt", None)
        if callable(broadcast):
            self.adapter.prepare_checkpoint()

    @contextlib.contextmanager
    def _swap_checkpoint_load_context(self, state_dict: Dict[str, Any], flatten: bool) -> Iterable[None]:
        """Let chain loading discover state keys without updating any parameters."""
        previous_active = self._export_active
        self._export_active = True
        try:
            with self.adapter.checkpoint_load_context(state_dict, flatten):
                yield
        finally:
            self._export_active = previous_active

    def _finish_swap_checkpoint(self) -> None:
        """Return checkpoint-broadcast state to its normal swapped form."""
        self.adapter.finish_checkpoint()

    def _broadcast_state_fused_for_ckpt(self) -> None:
        """Broadcast wrapped optimizer state through bounded device buffers."""
        broadcast = getattr(self.optimizer, "_broadcast_state_fused_for_ckpt", None)
        if not callable(broadcast):
            return
        prepared_here = not self.adapter._checkpoint_active
        try:
            if prepared_here:
                self.adapter.prepare_checkpoint()
            broadcast()
            self.adapter._checkpoint_broadcasted = True
        except Exception:
            if prepared_here:
                self.cleanup_synced_state()
            raise

    def cleanup_synced_state(self) -> None:
        """Release HSDP checkpoint copies and restore host-resident slots."""
        try:
            self.adapter.finish_checkpoint()
        finally:
            cleanup = getattr(self.optimizer, "cleanup_synced_state", None)
            if callable(cleanup):
                cleanup()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dict while keeping swappable tensors on CPU mirrors."""
        previous_active = self._export_active
        self._export_active = True
        try:
            self.adapter.load_checkpoint_state_dict(state_dict)
        finally:
            self._export_active = previous_active

    @contextlib.contextmanager
    def _no_grad_context(self) -> Iterable[None]:
        with torch.no_grad():
            yield
