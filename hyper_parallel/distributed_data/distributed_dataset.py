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
from hyper_parallel.distributed_data.fetcher import (
    LoadedSample,
    MetadataSource,
    MicroBatchFetcher,
    OnlineSampleSource,
    _pin_memory_batch,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import BatchPlan, DistributedDataStep, RankMicroBatch
from hyper_parallel.distributed_data.state import DatasetStateTracker
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.platform import get_platform

platform = get_platform()


@dataclass
class _ReservedStep:
    step: int
    cursor_start: int
    cursor_end: int
    plan: BatchPlan | None
    replay_id: str | None = None


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
    """Plan optimizer steps and fetch one local microbatch at a time.

    Sidecar metadata enables whole-step balancing before any sample read.
    Online metadata reads one global microbatch of candidates at a time and
    balances only within that microbatch. Call :meth:`commit` only after every
    microbatch and the corresponding optimizer step succeed.
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
        self._data_stream = data_stream
        source_size = len(metadata_source) if metadata_source is not None else len(online_sample_source)
        self._source_size = source_size
        self._state = DatasetStateTracker(
            source_size,
            planner.local_samples_per_step,
            prefetch_steps,
        )
        thread_name = "hp-data-buffer" if double_buffer else "hp-data-prefetch"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._pin_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hp-data-pin")
            if pin_memory and topology.is_data_owner
            else None
        )
        self._prepared_steps: deque[_ReservedStep] = deque()
        self._active_step: DistributedDataStep | None = None
        self._active_reservation: _ReservedStep | None = None
        self._owner_future: Future[Any] | None = None
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
            self._schedule_buffered_microbatch(reservation, 0)
        else:
            self._schedule_owner_microbatch(0)
        self._active_step = DistributedDataStep(
            step=reservation.step,
            cursor_start=reservation.cursor_start,
            cursor_end=reservation.cursor_end,
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
    def prepared_offset(self) -> int:
        """Return the offset reserved for bounded planning look-ahead."""
        return self._state.prepared_offset

    def commit(self, replay_id: str) -> None:
        """Commit one fully consumed step after optimizer-step success."""
        if self._active_step is None or not self._active_step.is_complete:
            raise ValueError("Consume every microbatch before committing the distributed-data step.")
        self._state.commit(replay_id)
        self._active_step = None
        self._active_reservation = None
        self._owner_future = None
        self._next_micro_batch_index = 0
        self._fill_prefetch()

    def state_dict(self) -> dict[str, int]:
        """Return checkpoint state at the successfully consumed offset."""
        return self._state.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the consumed offset before iteration starts."""
        if self._prepared_steps or self._active_step is not None:
            raise ValueError("Cannot restore distributed dataset state after data preparation has started.")
        self._state.load_state_dict(state_dict)

    def close(self) -> None:
        """Wait for current prefetch work and close the producer threads."""
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=not self._double_buffer)
        if self._pin_executor is not None:
            self._pin_executor.shutdown(wait=True, cancel_futures=True)
        self._prepared_steps.clear()
        self._active_step = None
        self._active_reservation = None
        self._owner_future = None
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
            step, cursor_start, cursor_end = self._state.reserve()
            plan = None
            if not self._double_buffer and self._topology.is_data_owner and self._metadata_source is not None:
                local_metadata = tuple(
                    self._metadata_source.get(index)
                    for index in range(cursor_start, cursor_end)
                )
                candidates = self._metadata_synchronizer.gather(
                    local_metadata,
                    self._topology.data_owner_ranks,
                )
                plan = self._planner.plan(candidates, step=step, cursor_start=cursor_start)
            self._prepared_steps.append(_ReservedStep(step, cursor_start, cursor_end, plan))

    def _schedule_owner_microbatch(self, micro_batch_index: int) -> None:
        self._owner_future = None
        if not self._topology.is_data_owner:
            return
        reservation = self._require_active_reservation()
        if self._metadata_source is not None:
            if reservation.plan is None:
                raise ValueError("Sidecar metadata did not produce a data-owner BatchPlan.")
            owner_future = self._executor.submit(
                self._micro_batch_fetcher.fetch,
                reservation.plan,
                self._topology.data_rank,
                micro_batch_index,
            )
            if self._pin_executor is not None:
                owner_future = self._pin_executor.submit(self._pin_fetched_microbatch, owner_future)
            self._owner_future = owner_future
            return

        cursor_start = reservation.cursor_start + micro_batch_index * self._planner.micro_batch_size
        cursor_end = cursor_start + self._planner.micro_batch_size
        self._owner_future = self._executor.submit(self._load_online_microbatch, cursor_start, cursor_end)

    def _load_online_microbatch(self, cursor_start: int, cursor_end: int) -> tuple[LoadedSample, ...]:
        if self._online_sample_source is None:
            raise ValueError("Online sample source is not configured.")
        return tuple(self._online_sample_source.get(index) for index in range(cursor_start, cursor_end))

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

        owner_input = None
        if self._topology.is_data_owner:
            if self._owner_future is None:
                raise ValueError("Data owner has no prepared microbatch future.")
            owner_input = self._owner_future.result()
        self._owner_future = None

        if self._metadata_source is not None:
            plan = reservation.plan if self._topology.is_data_owner else None
            owner_micro_batch = owner_input
        elif self._topology.is_data_owner:
            plan, owner_micro_batch = self._plan_online_microbatch(
                reservation,
                micro_batch_index,
                owner_input,
            )
        else:
            plan, owner_micro_batch = None, None

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
            replay_id=received_plan.replay_id,
            global_rank=self._topology.global_rank,
            data_rank=self._topology.data_rank,
            cp_rank=self._topology.cp_rank,
            micro_batch_index=micro_batch_index,
            sample_ids=tuple(sample.meta.sample_id for sample in planned_samples),
            data=rank_data,
        )

        self._next_micro_batch_index += 1
        if self._next_micro_batch_index < self._planner.micro_batch_num:
            self._schedule_owner_microbatch(self._next_micro_batch_index)
        return received_plan, micro_batch

    def _schedule_buffered_microbatch(self, reservation: _ReservedStep, micro_batch_index: int) -> None:
        key = (reservation.step, micro_batch_index)
        slot = self._buffer_slots[self._buffer_slot_index(reservation.step, micro_batch_index)]
        if slot.future is not None:
            if slot.key == key:
                return
            raise ValueError(f"Double-buffer slot {slot.index} still holds microbatch {slot.key}.")
        slot.key = key
        slot.future = self._executor.submit(
            self._prepare_buffered_microbatch,
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
        if self._next_micro_batch_index < self._planner.micro_batch_num:
            self._schedule_buffered_microbatch(reservation, self._next_micro_batch_index)
        elif self._prepared_steps:
            self._schedule_buffered_microbatch(self._prepared_steps[0], 0)
        return prepared.plan, prepared.micro_batch

    def _prepare_buffered_microbatch(
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
            plan, owner_micro_batch = self._prepare_buffered_owner_micro_batch(reservation, micro_batch_index)
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
                replay_id=received_plan.replay_id,
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

    def _prepare_buffered_owner_micro_batch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> tuple[BatchPlan | None, Any]:
        if not self._topology.is_data_owner:
            return None, None
        if self._metadata_source is not None:
            if reservation.plan is None:
                local_metadata = tuple(
                    self._metadata_source.get(index)
                    for index in range(reservation.cursor_start, reservation.cursor_end)
                )
                candidates = self._metadata_synchronizer.gather(
                    local_metadata,
                    self._topology.data_owner_ranks,
                )
                reservation.plan = self._planner.plan(
                    candidates,
                    step=reservation.step,
                    cursor_start=reservation.cursor_start,
                )
            owner_micro_batch = self._micro_batch_fetcher.fetch(
                reservation.plan,
                self._topology.data_rank,
                micro_batch_index,
            )
            if self._pin_executor is not None:
                owner_micro_batch = self._pin_executor.submit(_pin_memory_batch, owner_micro_batch).result()
            return reservation.plan, owner_micro_batch

        cursor_start = reservation.cursor_start + micro_batch_index * self._planner.micro_batch_size
        cursor_end = cursor_start + self._planner.micro_batch_size
        owner_input = self._load_online_microbatch(cursor_start, cursor_end)
        return self._plan_online_microbatch(reservation, micro_batch_index, owner_input)

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
        local_metadata = tuple(sample.metadata for sample in owner_input)
        candidates = self._metadata_synchronizer.gather(local_metadata, self._topology.data_owner_ranks)
        cursor_start = reservation.cursor_start + micro_batch_index * self._planner.micro_batch_size
        plan = self._planner.plan_microbatch(
            candidates,
            step=reservation.step,
            cursor_start=cursor_start,
            micro_batch_index=micro_batch_index,
        )
        samples_by_position = self._sample_redistributor.redistribute(
            tuple(sample.data for sample in owner_input),
            plan,
            self._topology,
        )
        planned_samples = plan.samples_for(self._topology.data_rank, micro_batch_index)
        samples = [samples_by_position[sample.source_position] for sample in planned_samples]
        owner_micro_batch = self._micro_batch_fetcher.collate(samples)
        if self._pin_executor is not None:
            owner_micro_batch = self._pin_executor.submit(_pin_memory_batch, owner_micro_batch).result()
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
            and plan.micro_batch_size == self._planner.micro_batch_size
        )
        if not dimensions_match:
            raise ValueError("Received BatchPlan dimensions do not match the distributed dataset planner.")
        if self._metadata_source is not None:
            expected_window = (reservation.cursor_start, reservation.cursor_end, 0, self._planner.micro_batch_num)
            if reservation.replay_id is None:
                reservation.replay_id = plan.replay_id
            elif plan.replay_id != reservation.replay_id:
                raise ValueError("Sidecar microbatches received inconsistent whole-step BatchPlan replay IDs.")
        else:
            cursor_start = reservation.cursor_start + micro_batch_index * self._planner.micro_batch_size
            expected_window = (cursor_start, cursor_start + self._planner.micro_batch_size, micro_batch_index, 1)
        actual_window = (plan.cursor_start, plan.cursor_end, plan.micro_batch_start, plan.micro_batch_num)
        if actual_window != expected_window:
            raise ValueError(f"Received BatchPlan window {actual_window} does not match expected {expected_window}.")

    def _complete_active_step(self, step: DistributedDataStep, replay_id: str) -> None:
        if step is not self._active_step:
            raise ValueError("Completed DistributedDataStep is not the dataset's active step.")
        reservation = self._require_active_reservation()
        if self._next_micro_batch_index != self._planner.micro_batch_num:
            raise ValueError("Cannot complete a DistributedDataStep before every microbatch is produced.")
        self._state.mark_delivered(replay_id, reservation.cursor_start, reservation.cursor_end)

    def _require_active_reservation(self) -> _ReservedStep:
        if self._active_reservation is None:
            raise ValueError("DistributedDataset has no active optimizer-step reservation.")
        return self._active_reservation

    @staticmethod
    def _pin_fetched_microbatch(owner_future: Future[Any]) -> Any:
        return _pin_memory_batch(owner_future.result())
