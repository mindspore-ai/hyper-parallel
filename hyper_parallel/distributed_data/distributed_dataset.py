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
"""Bounded-prefetch orchestration for planned distributed datasets."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from hyper_parallel.distributed_data.distributor import (
    MetadataSynchronizer,
    MicroBatchDistributor,
    SampleRedistributor,
)
from hyper_parallel.distributed_data.data_construct import (
    LoadedSample,
    LocalDataLoader,
    MetadataSource,
    MicroBatchFetcher,
    OnlineSampleSource,
    _pin_memory_batch,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import (
    BatchPlan,
    DistributedDataStep,
    OnlineSampleMetadata,
    RankMicroBatch,
)
from hyper_parallel.distributed_data.state import DatasetStateTracker
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.platform import get_platform

platform = get_platform()


@dataclass
class _ReservedStep:
    step: int
    sample_offset_start: int
    sample_offset_end: int
    host_future: Future["_HostReadyStep"] | None = None
    online_host_futures: dict[int, Future["_HostReadyMicroBatch"]] | None = None
    next_online_micro_batch_index: int = 0
    plan_id: str | None = None


@dataclass(frozen=True)
class _HostReadyStep:
    plans: tuple[BatchPlan | None, ...]
    micro_batches: tuple[Any, ...]


@dataclass(frozen=True)
class _HostReadyMicroBatch:
    plan: BatchPlan | None
    micro_batch: Any


@dataclass(frozen=True)
class _PreparedMicroBatch:
    plan: BatchPlan
    micro_batch: RankMicroBatch
    ready_event: Any = None


@dataclass
class _BufferSlot:
    index: int
    key: tuple[int, int] | None = None
    future: Future[_PreparedMicroBatch] | None = None


class DistributedDataset(Iterator[DistributedDataStep]):
    """Prefetch Host data and distribute one local microbatch at a time.

    Sidecar metadata enables whole-step balancing and Host prefetch. Online
    metadata keeps bounded microbatch-local reads, planning, and sample A2A.
    Device preparation and model-parallel distribution remain a separate
    one-microbatch stage. Call :meth:`commit` only after every microbatch and
    the corresponding optimizer step succeed.
    """

    def __init__(
        self,
        *,
        topology: DataTopology,
        metadata_source: MetadataSource | None,
        planner: DistributedBatchPlanner,
        micro_batch_fetcher: MicroBatchFetcher,
        metadata_synchronizer: MetadataSynchronizer,
        micro_batch_distributor: MicroBatchDistributor,
        prefetch_steps: int = 2,
        pin_memory: bool = False,
        double_buffer: bool = False,
        prepare_micro_batch: Callable[[Any], Any] | None = None,
        online_sample_source: OnlineSampleSource | None = None,
        sample_redistributor: SampleRedistributor | None = None,
        data_stream: Any = None,
        local_data_loader: LocalDataLoader | None = None,
    ) -> None:
        """Initialize planning, fetching, communication, and prefetch components."""
        if planner.data_parallel_size != topology.data_parallel_size:
            raise ValueError(
                f"Planner data_parallel_size={planner.data_parallel_size} does not match "
                f"topology data_parallel_size={topology.data_parallel_size}."
            )
        if (metadata_source is None) == (online_sample_source is None):
            raise ValueError("Configure exactly one of metadata_source and online_sample_source.")
        if online_sample_source is not None and sample_redistributor is None:
            raise ValueError("Online metadata requires a sample_redistributor.")
        if not isinstance(pin_memory, bool):
            raise ValueError(f"pin_memory must be a boolean, but got {pin_memory!r}.")
        if not isinstance(double_buffer, bool):
            raise ValueError(f"double_buffer must be a boolean, but got {double_buffer!r}.")
        self._topology = topology
        self._metadata_source = metadata_source
        self._online_sample_source = online_sample_source
        self._planner = planner
        self._micro_batch_fetcher = micro_batch_fetcher
        self._metadata_synchronizer = metadata_synchronizer
        self._sample_redistributor = sample_redistributor
        self._micro_batch_distributor = micro_batch_distributor
        self._prepare_micro_batch = prepare_micro_batch
        self._double_buffer = double_buffer
        self._prefetch_steps = prefetch_steps
        self._data_stream = data_stream
        self._local_data_loader = local_data_loader
        source_size = len(metadata_source) if metadata_source is not None else len(online_sample_source)
        self._source_size = source_size
        self._state = DatasetStateTracker(
            source_size,
            planner.local_samples_per_step,
            prefetch_steps,
        )
        producer_thread_name = "hp-data-buffer" if double_buffer else "hp-data-producer"
        self._producer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=producer_thread_name)
        self._host_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hp-data-host")
        self._pin_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hp-data-pin")
            if pin_memory and topology.is_data_owner and not micro_batch_fetcher.fetched_batches_are_pinned
            else None
        )
        self._prepared_steps: deque[_ReservedStep] = deque()
        self._online_pending_units: deque[tuple[int, int]] = deque()
        self._active_step: DistributedDataStep | None = None
        self._active_reservation: _ReservedStep | None = None
        self._active_future: Future[_PreparedMicroBatch] | None = None
        self._buffer_slots = [_BufferSlot(0), _BufferSlot(1)] if double_buffer else []
        self._next_micro_batch_index = 0
        self._closed = False

    def __len__(self) -> int:
        """Return complete optimizer steps in this rank's candidate shard."""
        return self._source_size // self._planner.local_samples_per_step

    def __iter__(self) -> "DistributedDataset":
        """Return this stateful optimizer-step iterator."""
        return self

    def __next__(self) -> DistributedDataStep:
        """Return the next lazy optimizer-step iterator."""
        if self._closed:
            raise StopIteration
        if self._active_step is not None:
            raise ValueError("Fully consume and commit the current DistributedDataStep before requesting another.")
        self._fill_prefetch()
        if not self._prepared_steps:
            raise StopIteration
        reservation = self._prepared_steps.popleft()
        self._active_reservation = reservation
        self._next_micro_batch_index = 0
        if self._double_buffer:
            self._ensure_online_host_microbatch(reservation, 0)
            self._schedule_buffered_microbatch(reservation, 0)
        else:
            self._ensure_online_host_microbatch(reservation, 0)
        self._fill_online_host_prefetch()
        self._active_step = DistributedDataStep(
            step=reservation.step,
            sample_offset_start=reservation.sample_offset_start,
            sample_offset_end=reservation.sample_offset_end,
            micro_batch_num=self._planner.micro_batch_num,
            load_micro_batch=self._consume_active_microbatch,
            on_complete=self._complete_active_step,
        )
        return self._active_step

    @property
    def consumed_offset(self) -> int:
        """Return the offset after the last successful optimizer step."""
        return self._state.consumed_offset

    @property
    def reserved_offset(self) -> int:
        """Return the offset after all scheduled Host-prefetch windows."""
        return self._state.reserved_offset

    @property
    def ready_offset(self) -> int:
        """Return the offset after contiguous steps ready in Host memory."""
        return self._state.ready_offset

    def commit(self, plan_id: str) -> None:
        """Commit one fully consumed step after optimizer-step success.

        Args:
            plan_id: Deterministic identifier returned by the consumed step.
        """
        if self._active_step is None or not self._active_step.is_complete:
            raise ValueError("Consume every microbatch before committing the distributed-data step.")
        self._state.commit(plan_id)
        self._active_step = None
        self._active_reservation = None
        self._active_future = None
        self._next_micro_batch_index = 0
        self._fill_prefetch()

    def state_dict(self) -> dict[str, int]:
        """Return checkpoint state at the successfully consumed offset."""
        return self._state.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the consumed offset before iteration starts.

        Args:
            state_dict: Checkpoint state created by :meth:`state_dict`.
        """
        if self._prepared_steps or self._active_step is not None:
            raise ValueError("Cannot restore distributed dataset state after data preparation has started.")
        self._state.load_state_dict(state_dict)

    def close(self) -> None:
        """Wait for current prefetch work and close the producer threads."""
        if self._closed:
            return
        # Host tasks and data collectives may wait on each other. Keep both
        # executors alive while draining, then release their threads.
        self._host_executor.shutdown(wait=True, cancel_futures=False)
        self._producer_executor.shutdown(wait=True, cancel_futures=False)
        if self._pin_executor is not None:
            self._pin_executor.shutdown(wait=True, cancel_futures=True)
        if self._local_data_loader is not None:
            self._local_data_loader.close()
        self._prepared_steps.clear()
        self._online_pending_units.clear()
        self._active_step = None
        self._active_reservation = None
        self._active_future = None
        for slot in self._buffer_slots:
            slot.key = None
            slot.future = None
        self._closed = True

    def __enter__(self) -> "DistributedDataset":
        """Enter a context that closes prefetch resources on exit."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close prefetch resources when leaving a context."""
        self.close()

    def _fill_prefetch(self) -> None:
        while self._state.can_prefetch and self._state.can_reserve_full_step():
            step, sample_offset_start, sample_offset_end = self._state.reserve()
            reservation = _ReservedStep(step, sample_offset_start, sample_offset_end)
            if self._metadata_source is not None:
                plan_future = self._producer_executor.submit(self._plan_sidecar_step, reservation)
                reservation.host_future = self._host_executor.submit(
                    self._prepare_sidecar_host_step,
                    reservation,
                    plan_future,
                )
            else:
                reservation.online_host_futures = {}
            self._prepared_steps.append(reservation)

    def _plan_sidecar_step(self, reservation: _ReservedStep) -> BatchPlan | None:
        if not self._topology.is_data_owner:
            return None
        if self._metadata_source is None:
            raise ValueError("Sidecar metadata source is not configured.")
        local_metadata = tuple(
            self._metadata_source.get(index)
            for index in range(reservation.sample_offset_start, reservation.sample_offset_end)
        )
        candidates = self._metadata_synchronizer.gather(
            local_metadata,
            self._topology.data_owner_ranks,
        )
        return self._planner.plan(
            candidates,
            step=reservation.step,
            sample_offset_start=reservation.sample_offset_start,
        )

    def _prepare_sidecar_host_step(
        self,
        reservation: _ReservedStep,
        plan_future: Future[BatchPlan | None],
    ) -> _HostReadyStep:
        plan = plan_future.result()
        micro_batches = []
        for micro_batch_index in range(self._planner.micro_batch_num):
            micro_batch = None
            if self._topology.is_data_owner:
                if plan is None:
                    raise ValueError("Sidecar metadata did not produce a data-owner BatchPlan.")
                micro_batch = self._micro_batch_fetcher.fetch(
                    plan,
                    self._topology.data_rank,
                    micro_batch_index,
                )
                micro_batch = self._pin_host_batch(micro_batch)
            micro_batches.append(micro_batch)
        self._state.mark_ready(reservation.sample_offset_start, reservation.sample_offset_end)
        return _HostReadyStep(
            plans=tuple(plan for _ in range(self._planner.micro_batch_num)),
            micro_batches=tuple(micro_batches),
        )

    def _prepare_online_host_microbatch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
        loaded_future: Future[tuple[LoadedSample, ...] | None],
    ) -> _HostReadyMicroBatch:
        owner_input = loaded_future.result()
        if self._topology.is_data_owner:
            plan, micro_batch = self._plan_online_microbatch(
                reservation,
                micro_batch_index,
                owner_input,
            )
        else:
            plan, micro_batch = None, None
        if micro_batch_index + 1 == self._planner.micro_batch_num:
            self._state.mark_ready(reservation.sample_offset_start, reservation.sample_offset_end)
        return _HostReadyMicroBatch(plan, micro_batch)

    def _ensure_online_host_microbatch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> None:
        if self._metadata_source is not None:
            return
        if reservation.online_host_futures is None:
            raise ValueError("Online reservation has no Host-prefetch future table.")
        if micro_batch_index in reservation.online_host_futures:
            return
        if micro_batch_index != reservation.next_online_micro_batch_index:
            raise ValueError(
                f"Expected online Host-prefetch microbatch {reservation.next_online_micro_batch_index}, "
                f"but got {micro_batch_index}."
            )
        sample_offset_start = (
            reservation.sample_offset_start + micro_batch_index * self._planner.raw_sample_size
        )
        sample_offset_end = sample_offset_start + self._planner.raw_sample_size
        if self._topology.is_data_owner:
            loaded_future = self._host_executor.submit(
                self._load_online_microbatch,
                sample_offset_start,
                sample_offset_end,
            )
        else:
            loaded_future = Future()
            loaded_future.set_result(None)
        reservation.online_host_futures[micro_batch_index] = self._producer_executor.submit(
            self._prepare_online_host_microbatch,
            reservation,
            micro_batch_index,
            loaded_future,
        )
        reservation.next_online_micro_batch_index += 1
        self._online_pending_units.append((reservation.step, micro_batch_index))

    def _fill_online_host_prefetch(self) -> None:
        if self._metadata_source is not None:
            return
        reservations = []
        if self._active_reservation is not None:
            reservations.append(self._active_reservation)
        reservations.extend(self._prepared_steps)
        for reservation in reservations:
            while (
                len(self._online_pending_units) < self._prefetch_steps
                and reservation.next_online_micro_batch_index < self._planner.micro_batch_num
            ):
                self._ensure_online_host_microbatch(
                    reservation,
                    reservation.next_online_micro_batch_index,
                )
            if len(self._online_pending_units) >= self._prefetch_steps:
                return

    def _load_online_microbatch(
        self,
        sample_offset_start: int,
        sample_offset_end: int,
    ) -> tuple[LoadedSample, ...]:
        if self._online_sample_source is None:
            raise ValueError("Online sample source is not configured.")
        return self._online_sample_source.get_range(sample_offset_start, sample_offset_end)

    def _consume_active_microbatch(self, micro_batch_index: int) -> tuple[BatchPlan, RankMicroBatch]:
        if self._closed:
            raise ValueError("Cannot consume microbatches from a closed DistributedDataset.")
        reservation = self._require_active_reservation()
        if micro_batch_index != self._next_micro_batch_index:
            raise ValueError(
                f"Expected microbatch {self._next_micro_batch_index}, but got {micro_batch_index}."
            )
        if self._double_buffer:
            return self._consume_buffered_microbatch(reservation, micro_batch_index)
        if self._active_future is None:
            self._schedule_active_microbatch(reservation, micro_batch_index)
        if self._active_future is None:
            raise ValueError("Distributed-data producer did not schedule the active microbatch.")
        prepared = self._active_future.result()
        self._active_future = None
        self._next_micro_batch_index += 1
        self._finish_online_host_microbatch(reservation, micro_batch_index)
        return prepared.plan, prepared.micro_batch

    def _schedule_active_microbatch(self, reservation: _ReservedStep, micro_batch_index: int) -> None:
        if self._active_future is not None:
            raise ValueError("The active distributed-data microbatch is already scheduled.")
        self._active_future = self._producer_executor.submit(
            self._prepare_distributed_microbatch,
            reservation,
            micro_batch_index,
        )

    def _schedule_buffered_microbatch(self, reservation: _ReservedStep, micro_batch_index: int) -> None:
        key = (reservation.step, micro_batch_index)
        slot = self._buffer_slots[self._buffer_slot_index(reservation.step, micro_batch_index)]
        if slot.future is not None:
            if slot.key == key:
                return
            raise ValueError(f"Double-buffer slot {slot.index} still holds microbatch {slot.key}.")
        slot.key = key
        slot.future = self._producer_executor.submit(
            self._prepare_distributed_microbatch,
            reservation,
            micro_batch_index,
        )

    def _consume_buffered_microbatch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> tuple[BatchPlan, RankMicroBatch]:
        key = (reservation.step, micro_batch_index)
        slot = self._buffer_slots[self._buffer_slot_index(reservation.step, micro_batch_index)]
        if slot.key != key or slot.future is None:
            raise ValueError(f"Double-buffer slot {slot.index} does not contain expected microbatch {key}.")
        try:
            prepared = slot.future.result()
        finally:
            slot.key = None
            slot.future = None
        if prepared.ready_event is not None:
            prepared.ready_event.wait(platform.get_current_stream())

        self._next_micro_batch_index += 1
        self._finish_online_host_microbatch(reservation, micro_batch_index)
        if self._next_micro_batch_index < self._planner.micro_batch_num:
            self._schedule_buffered_microbatch(reservation, self._next_micro_batch_index)
        elif self._prepared_steps:
            self._schedule_buffered_microbatch(self._prepared_steps[0], 0)
        return prepared.plan, prepared.micro_batch

    def _prepare_distributed_microbatch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> _PreparedMicroBatch:
        stream_context = (
            nullcontext()
            if self._data_stream is None
            else platform.get_stream_context()(self._data_stream)
        )
        with stream_context:
            if self._metadata_source is not None:
                if reservation.host_future is None:
                    raise ValueError("Reserved sidecar step has no Host-prefetch future.")
                host_step = reservation.host_future.result()
                plan = host_step.plans[micro_batch_index]
                owner_micro_batch = host_step.micro_batches[micro_batch_index]
            else:
                if (
                    reservation.online_host_futures is None
                    or micro_batch_index not in reservation.online_host_futures
                ):
                    raise ValueError("Online microbatch has no Host-prefetch future.")
                host_micro_batch = reservation.online_host_futures[micro_batch_index].result()
                plan = host_micro_batch.plan
                owner_micro_batch = host_micro_batch.micro_batch
            if owner_micro_batch is not None and self._prepare_micro_batch is not None:
                owner_micro_batch = self._prepare_micro_batch(owner_micro_batch)
            received_plan, rank_data = self._micro_batch_distributor.distribute(
                owner_micro_batch,
                plan,
                self._topology,
            )
            self._validate_received_plan(received_plan, reservation, micro_batch_index)
            planned_samples = received_plan.samples_for(self._topology.data_rank, micro_batch_index)
            micro_batch = RankMicroBatch(
                plan_id=received_plan.plan_id,
                global_rank=self._topology.global_rank,
                data_rank=self._topology.data_rank,
                cp_rank=self._topology.cp_rank,
                micro_batch_index=micro_batch_index,
                sample_ids=tuple(sample.meta.sample_id for sample in planned_samples),
                data=rank_data,
            )
            ready_event = None
            if self._data_stream is not None:
                ready_event = platform.new_event()
                ready_event.record(self._data_stream)
        return _PreparedMicroBatch(received_plan, micro_batch, ready_event)

    def _buffer_slot_index(self, step: int, micro_batch_index: int) -> int:
        sequence = step * self._planner.micro_batch_num + micro_batch_index
        return sequence % len(self._buffer_slots)

    def _plan_online_microbatch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
        owner_input: Any,
    ) -> tuple[BatchPlan, Any]:
        if self._sample_redistributor is None:
            raise ValueError("Online metadata requires a sample redistributor.")
        if not isinstance(owner_input, tuple) or any(not isinstance(sample, LoadedSample) for sample in owner_input):
            raise ValueError("Online microbatch preparation returned invalid loaded samples.")
        local_metadata = tuple(
            OnlineSampleMetadata(
                sample_meta=sample.metadata,
                tensor_spec=self._sample_redistributor.describe_sample(sample.data),
            )
            for sample in owner_input
        )
        global_metadata = self._metadata_synchronizer.gather(local_metadata, self._topology.data_owner_ranks)
        if any(not isinstance(metadata, OnlineSampleMetadata) for metadata in global_metadata):
            raise ValueError("Online metadata synchronization returned invalid sample transport metadata.")
        candidates = tuple(metadata.sample_meta for metadata in global_metadata)
        sample_offset_start = (
            reservation.sample_offset_start + micro_batch_index * self._planner.raw_sample_size
        )
        plan = self._planner.plan_microbatch(
            candidates,
            step=reservation.step,
            sample_offset_start=sample_offset_start,
            micro_batch_index=micro_batch_index,
        )
        samples_by_position = self._sample_redistributor.redistribute(
            tuple(sample.data for sample in owner_input),
            plan,
            self._topology,
            global_metadata,
        )
        planned_samples = plan.samples_for(self._topology.data_rank, micro_batch_index)
        samples = [samples_by_position[sample.source_position] for sample in planned_samples]
        owner_micro_batch = self._micro_batch_fetcher.collate(samples)
        owner_micro_batch = self._pin_host_batch(owner_micro_batch)
        return plan, owner_micro_batch

    def _validate_received_plan(
        self,
        plan: BatchPlan,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> None:
        if plan.step != reservation.step:
            raise ValueError(f"Expected optimizer step {reservation.step}, but received plan step {plan.step}.")
        dimensions_match = (
            plan.data_parallel_size == self._planner.data_parallel_size
            and plan.raw_sample_size == self._planner.raw_sample_size
        )
        if not dimensions_match:
            raise ValueError("Received BatchPlan dimensions do not match the distributed dataset planner.")
        if self._metadata_source is not None:
            expected_window = (
                reservation.sample_offset_start,
                reservation.sample_offset_end,
                0,
                self._planner.micro_batch_num,
            )
            if reservation.plan_id is None:
                reservation.plan_id = plan.plan_id
            elif plan.plan_id != reservation.plan_id:
                raise ValueError("Sidecar microbatches received inconsistent whole-step plan IDs.")
        else:
            sample_offset_start = (
                reservation.sample_offset_start + micro_batch_index * self._planner.raw_sample_size
            )
            expected_window = (
                sample_offset_start,
                sample_offset_start + self._planner.raw_sample_size,
                micro_batch_index,
                1,
            )
        actual_window = (
            plan.sample_offset_start,
            plan.sample_offset_end,
            plan.micro_batch_start,
            plan.micro_batch_num,
        )
        if actual_window != expected_window:
            raise ValueError(f"Received BatchPlan window {actual_window} does not match expected {expected_window}.")

    def _complete_active_step(self, step: DistributedDataStep, plan_id: str) -> None:
        if step is not self._active_step:
            raise ValueError("Completed DistributedDataStep is not the dataset's active step.")
        reservation = self._require_active_reservation()
        if self._next_micro_batch_index != self._planner.micro_batch_num:
            raise ValueError("Cannot complete a DistributedDataStep before every microbatch is produced.")
        self._state.mark_delivered(
            plan_id,
            reservation.sample_offset_start,
            reservation.sample_offset_end,
        )
        self._fill_prefetch()
        self._fill_online_host_prefetch()

    def _require_active_reservation(self) -> _ReservedStep:
        if self._active_reservation is None:
            raise ValueError("DistributedDataset has no active optimizer-step reservation.")
        return self._active_reservation

    def _pin_host_batch(self, batch: Any) -> Any:
        if self._pin_executor is None:
            return batch
        return self._pin_executor.submit(_pin_memory_batch, batch).result()

    def _finish_online_host_microbatch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> None:
        if self._metadata_source is not None:
            return
        expected_key = (reservation.step, micro_batch_index)
        if not self._online_pending_units or self._online_pending_units[0] != expected_key:
            raise ValueError(f"Online Host-prefetch units must be consumed in order; expected {expected_key}.")
        self._online_pending_units.popleft()
        self._fill_online_host_prefetch()
