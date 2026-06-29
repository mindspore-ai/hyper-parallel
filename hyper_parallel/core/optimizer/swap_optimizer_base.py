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
"""Common abstractions for optimizer state swap."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
MASTER_PARAM_KEY = "master_param"
SUPPORTED_STATE_KEYS = STATE_KEYS + (MASTER_PARAM_KEY,)


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


@dataclass
class UpdateUnit:
    """Per-parameter optimizer update unit used by the pipeline runtime.

    ``adapter_index`` is interpreted by the backend adapter: it identifies a
    Torch parameter group or a MindSpore parameter/state entry.
    """

    adapter_index: int
    param: Any
    grad: Any
    slots: List[SwapSlot]


class OptimizerSwapAdapter:
    """Backend optimizer adapter interface."""

    def __init__(self, optimizer: Any, config: Any, runtime: Any) -> None:
        self.optimizer = optimizer
        self.config = config
        self.runtime = runtime

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        """Return whether this adapter supports ``optimizer``."""
        raise NotImplementedError

    def validate(self) -> None:
        """Validate optimizer flags and unsupported modes."""
        raise NotImplementedError

    def prepare_step(self, *args: Any, **kwargs: Any) -> Any:
        """Prepare one outer optimizer step."""
        raise NotImplementedError

    def iter_update_units(self, step_context: Any) -> List[UpdateUnit]:
        """Return update units for the current outer step."""
        raise NotImplementedError

    def step_batch(self, batch: List[UpdateUnit], step_context: Any) -> Any:
        """Update one pipeline batch."""
        raise NotImplementedError

    def finish_step(self, step_context: Any) -> Any:
        """Finish one outer optimizer step."""
        del step_context

    def all_slots(self) -> Iterable[SwapSlot]:
        """Iterate all known swap slots."""
        return ()

    def initial_slots(self) -> Iterable[SwapSlot]:
        """Build or return slots that can be offloaded before the first update."""
        return self.all_slots()

    def checkpoint_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return a checkpoint view when the backend supports it."""
        raise NotImplementedError

    def load_checkpoint_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> None:
        """Load a checkpoint view when the backend supports it."""
        raise NotImplementedError


class PipelineSwapRuntime:
    """Backend-neutral pipeline orchestration.

    Concrete backend runtimes provide tensor copy/storage methods. This class
    owns batching and the common state machine order.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._copy_stream: Optional[Any] = None

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
            max_end = len(non_empty_units) - remaining_batches + 1
            end = start
            current_cost = 0
            while end < max_end:
                next_cost = unit_costs[end]
                # Compare against the remaining average without introducing floating-point rounding.
                current_distance = abs(remaining_cost - current_cost * remaining_batches)
                next_distance = abs(remaining_cost - (current_cost + next_cost) * remaining_batches)
                if end > start and current_distance <= next_distance:
                    break
                current_cost += next_cost
                end += 1
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
        return results

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

    def release_packed_step_results(self, results: List[Any]) -> None:
        """Release backend-specific update outputs before staging teardown."""
        del results

    def supports_packed_pipeline(self, batches: Sequence[Sequence[UpdateUnit]]) -> bool:
        """Return whether this runtime can use two packed staging buffers."""
        del batches
        return False

    def begin_packed_step(self, batches: Sequence[Sequence[UpdateUnit]]) -> None:
        """Allocate and prepare packed staging storage for one optimizer step."""
        del batches
        raise NotImplementedError

    def enqueue_packed_prefetch(self, batch_index: int, staging_index: int) -> None:
        """Enqueue a standalone H2D into one staging buffer."""
        del batch_index, staging_index
        raise NotImplementedError

    def wait_packed_prefetch(self, batch_index: int, staging_index: int) -> None:
        """Make compute wait until a packed batch is ready for update."""
        del batch_index, staging_index
        raise NotImplementedError

    def activate_packed_batch(self, batch_index: int, staging_index: int) -> None:
        """Bind batch optimizer states to views in a staging buffer."""
        del batch_index, staging_index
        raise NotImplementedError

    def enqueue_packed_offload_prefetch(
            self,
            batch_index: int,
            next_index: Optional[int],
            staging_index: int,
    ) -> None:
        """Enqueue D2H and the next H2D serially through one staging buffer."""
        del batch_index, next_index, staging_index
        raise NotImplementedError

    def wait_packed_offload(self, batch_index: int) -> None:
        """Synchronize a trailing D2H before staging storage is released."""
        del batch_index
        raise NotImplementedError

    def finish_packed_offload(self, batch_index: int) -> None:
        """Rebind an offloaded batch to its persistent CPU state views."""
        del batch_index
        raise NotImplementedError

    def end_packed_step(self) -> None:
        """Finish backend staging cleanup after all transfers complete."""
        raise NotImplementedError

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

    def _unit_cost(self, unit: UpdateUnit) -> int:
        return sum(slot.storage_nbytes for slot in unit.slots if slot.swappable)

    def _get_copy_stream(self) -> Any:
        if self._copy_stream is None:
            self._copy_stream = self.new_stream()
        return self._copy_stream

    def _record_current_stream_event(self) -> Any:
        current_stream = self.current_stream()
        return self.record_event(current_stream)

    def current_stream(self) -> Any:
        """Return the current compute stream for the active backend."""
        raise NotImplementedError

    def new_stream(self) -> Any:
        """Create a copy stream for the active backend."""
        raise NotImplementedError

    def stream_context(self, stream: Any):
        """Return a context manager that makes ``stream`` current."""
        del stream
        return contextlib.nullcontext()

    def restore_device_storage(self, slot: SwapSlot) -> None:
        """Restore device storage before enqueueing H2D copy."""
        del slot

    def record_event(self, stream: Any = None) -> Any:
        """Record an event on ``stream`` when the backend supports events."""
        del stream

    def wait_event(self, event: Any, stream: Any = None) -> None:
        """Make ``stream`` wait for ``event`` when the backend supports events."""
        del event, stream

    def make_cpu_tensor(self, tensor: Any) -> Any:
        """Create a CPU mirror tensor."""
        raise NotImplementedError

    def copy_to_device(self, slot: SwapSlot) -> None:
        """Copy one slot CPU mirror to its device tensor."""
        raise NotImplementedError

    def wait_prefetch_slot(self, slot: SwapSlot) -> None:
        """Wait for one slot prefetch."""
        raise NotImplementedError

    def copy_to_cpu(self, slot: SwapSlot) -> None:
        """Copy one slot device tensor to its CPU mirror."""
        raise NotImplementedError

    def wait_offload_slot(self, slot: SwapSlot) -> None:
        """Wait for one slot offload."""
        raise NotImplementedError


def _iter_unique_slots(units: Sequence[UpdateUnit]) -> Iterable[SwapSlot]:
    """Yield slots once by object identity."""
    return _iter_unique_slot_objects(slot for unit in units for slot in unit.slots)


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


def validate_state_keys(state_keys: Optional[Sequence[str]]) -> Optional[tuple[str, ...]]:
    """Validate user-provided logical state keys."""
    if state_keys is None:
        return None
    normalized = tuple(state_keys)
    invalid = sorted(set(normalized) - set(SUPPORTED_STATE_KEYS))
    if invalid:
        raise ValueError(
            "SwapOptimizerConfig.state_keys only supports Adam/AdamW logical slots "
            f"{SUPPORTED_STATE_KEYS}, but got {invalid}."
        )
    return normalized
