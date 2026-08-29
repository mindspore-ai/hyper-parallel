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
"""Bounded-prefetch orchestration for planned local batches."""
# This package is intentionally PyTorch-only.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from hyper_parallel.distributed_data.data_construct import (
    LoadedLocalBatch,
    LocalBatchMetadataView,
    LocalBatchSource,
    OnlineLocalBatchView,
    RankLocalDataLoaderSource,
    SidecarLocalBatchFetcher,
)
from hyper_parallel.distributed_data.distributor import (
    LocalBatchRedistributor,
    MetadataSynchronizer,
    ModelParallelLocalBatchDistributor,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import (
    BatchPlan,
    DistributedDataStep,
    LocalBatch,
    OnlineLocalBatchMetadata,
)
from hyper_parallel.distributed_data.state import DatasetStateTracker
from hyper_parallel.distributed_data.topology import DataTopology


def _stream_device_module(device: Any) -> Any:
    """Return the PyTorch accelerator module for a stream device."""
    device_type = torch.device(device).type
    device_module = getattr(torch, device_type, None)
    if device_module is None:
        raise ValueError(f"PyTorch has no device module for data stream device {device!r}.")
    return device_module


@dataclass
class _ReservedStep:
    step: int
    local_batch_offset_start: int
    local_batch_offset_end: int
    host_future: Future["_HostReadyStep"] | None = None
    plan_id: str | None = None


@dataclass(frozen=True)
class _HostReadyStep:
    plan: BatchPlan | None
    local_batches: tuple[Any, ...]


@dataclass(frozen=True)
class _PreparedLocalBatch:
    plan: BatchPlan
    local_batch: LocalBatch
    ready_event: Any = None


@dataclass
class _BufferSlot:
    index: int
    key: tuple[int, int] | None = None
    future: Future[_PreparedLocalBatch] | None = None


class DistributedDataset(Iterator[DistributedDataStep]):
    """Plan, prefetch, and distribute complete single-card local batches."""

    def __init__(
        self,
        *,
        topology: DataTopology,
        source: LocalBatchSource,
        metadata_source: LocalBatchMetadataView | None,
        online_source: OnlineLocalBatchView | RankLocalDataLoaderSource | None,
        planner: DistributedBatchPlanner,
        sidecar_fetcher: SidecarLocalBatchFetcher | None,
        metadata_synchronizer: MetadataSynchronizer,
        local_batch_redistributor: LocalBatchRedistributor | None,
        model_parallel_distributor: ModelParallelLocalBatchDistributor,
        prefetch_steps: int = 2,
        double_buffer: bool = False,
        prepare_local_batch: Callable[[Any], Any] | None = None,
        data_stream: Any = None,
        data_stream_device: Any = None,
    ) -> None:
        """Initialize local-batch planning, fetching, communication, and prefetch."""
        if planner.data_parallel_size != topology.data_parallel_size:
            raise ValueError(
                f"Planner data_parallel_size={planner.data_parallel_size} does not match "
                f"topology data_parallel_size={topology.data_parallel_size}."
            )
        if (metadata_source is None) == (online_source is None):
            raise ValueError("Configure exactly one of metadata_source and online_source.")
        if metadata_source is not None and sidecar_fetcher is None:
            raise ValueError("Sidecar metadata requires a sidecar_fetcher.")
        if online_source is not None and local_batch_redistributor is None:
            raise ValueError("Online local batches require a local_batch_redistributor.")
        if not isinstance(prefetch_steps, int) or isinstance(prefetch_steps, bool) or prefetch_steps < 1:
            raise ValueError(f"prefetch_steps must be a positive integer, but got {prefetch_steps!r}.")
        if not isinstance(double_buffer, bool):
            raise ValueError(f"double_buffer must be a boolean, but got {double_buffer!r}.")
        if prepare_local_batch is not None and not callable(prepare_local_batch):
            raise ValueError("prepare_local_batch must be callable or None.")
        if (data_stream is None) != (data_stream_device is None):
            raise ValueError("data_stream and data_stream_device must be configured together.")

        self._topology = topology
        self._source = source
        self._metadata_source = metadata_source
        self._online_source = online_source
        self._planner = planner
        self._sidecar_fetcher = sidecar_fetcher
        self._metadata_synchronizer = metadata_synchronizer
        self._local_batch_redistributor = local_batch_redistributor
        self._model_parallel_distributor = model_parallel_distributor
        self._prepare_local_batch = prepare_local_batch
        self._double_buffer = double_buffer
        self._data_stream = data_stream
        self._data_stream_device = data_stream_device
        source_size = len(metadata_source) if metadata_source is not None else len(online_source)
        self._source_size = source_size
        self._state = DatasetStateTracker(source_size, planner.local_batches_per_step, prefetch_steps)
        producer_name = "hp-data-buffer" if double_buffer else "hp-data-producer"
        self._producer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=producer_name)
        self._host_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hp-data-host")
        self._prepared_steps: deque[_ReservedStep] = deque()
        self._active_step: DistributedDataStep | None = None
        self._active_reservation: _ReservedStep | None = None
        self._active_future: Future[_PreparedLocalBatch] | None = None
        self._buffer_slots = [_BufferSlot(0), _BufferSlot(1)] if double_buffer else []
        self._next_micro_batch_index = 0
        self._closed = False

    def __len__(self) -> int:
        """Return complete optimizer steps in this rank's source shard."""
        return self._source_size // self._planner.local_batches_per_step

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
            self._schedule_buffered_local_batch(reservation, 0)
        self._active_step = DistributedDataStep(
            step=reservation.step,
            local_batch_offset_start=reservation.local_batch_offset_start,
            local_batch_offset_end=reservation.local_batch_offset_end,
            micro_batch_num=self._planner.micro_batch_num,
            load_local_batch=self._consume_active_local_batch,
            on_complete=self._complete_active_step,
        )
        return self._active_step

    @property
    def consumed_offset(self) -> int:
        """Return the local-batch offset after the last successful step."""
        return self._state.consumed_offset

    @property
    def reserved_offset(self) -> int:
        """Return the offset after all scheduled Host-prefetch windows."""
        return self._state.reserved_offset

    @property
    def ready_offset(self) -> int:
        """Return the offset after contiguous Host-ready steps."""
        return self._state.ready_offset

    def commit(self, plan_id: str) -> None:
        """Commit one fully consumed step after optimizer-step success."""
        if self._active_step is None or not self._active_step.is_complete:
            raise ValueError("Consume every local batch before committing the distributed-data step.")
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
        """Restore consumed state before iteration starts."""
        if self._prepared_steps or self._active_step is not None:
            raise ValueError("Cannot restore distributed dataset state after data preparation has started.")
        self._state.load_state_dict(state_dict)

    def close(self) -> None:
        """Drain prefetch work and close source resources."""
        if self._closed:
            return
        self._host_executor.shutdown(wait=True, cancel_futures=False)
        self._producer_executor.shutdown(wait=True, cancel_futures=False)
        self._source.close()
        self._prepared_steps.clear()
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
            step, offset_start, offset_end = self._state.reserve()
            reservation = _ReservedStep(step, offset_start, offset_end)
            if self._metadata_source is not None:
                plan_future = self._producer_executor.submit(self._plan_sidecar_step, reservation)
                reservation.host_future = self._host_executor.submit(
                    self._fetch_sidecar_step,
                    reservation,
                    plan_future,
                )
            else:
                loaded_future = self._host_executor.submit(self._load_online_step, reservation)
                reservation.host_future = self._producer_executor.submit(
                    self._prepare_online_step,
                    reservation,
                    loaded_future,
                )
            self._prepared_steps.append(reservation)

    def _plan_sidecar_step(self, reservation: _ReservedStep) -> BatchPlan | None:
        if not self._topology.is_data_owner:
            return None
        if self._metadata_source is None:
            raise ValueError("Sidecar metadata source is not configured.")
        local_metadata = tuple(
            self._metadata_source.get(index)
            for index in range(
                reservation.local_batch_offset_start,
                reservation.local_batch_offset_end,
            )
        )
        candidates = self._metadata_synchronizer.gather(local_metadata, self._topology.data_owner_ranks)
        return self._planner.plan(
            candidates,
            step=reservation.step,
            local_batch_offset_start=reservation.local_batch_offset_start,
        )

    def _fetch_sidecar_step(
        self,
        reservation: _ReservedStep,
        plan_future: Future[BatchPlan | None],
    ) -> _HostReadyStep:
        plan = plan_future.result()
        local_batches = []
        for micro_batch_index in range(self._planner.micro_batch_num):
            local_batch = None
            if self._topology.is_data_owner:
                if plan is None or self._sidecar_fetcher is None:
                    raise ValueError("Sidecar planning did not produce a data-owner plan and fetcher.")
                local_batch = self._sidecar_fetcher.fetch(
                    plan,
                    self._topology.data_rank,
                    micro_batch_index,
                )
            local_batches.append(local_batch)
        self._state.mark_ready(
            reservation.local_batch_offset_start,
            reservation.local_batch_offset_end,
        )
        return _HostReadyStep(plan, tuple(local_batches))

    def _load_online_step(self, reservation: _ReservedStep) -> tuple[LoadedLocalBatch, ...] | None:
        if not self._topology.is_data_owner:
            return None
        if self._online_source is None:
            raise ValueError("Online local-batch source is not configured.")
        return self._online_source.get_range(
            reservation.local_batch_offset_start,
            reservation.local_batch_offset_end,
        )

    def _prepare_online_step(
        self,
        reservation: _ReservedStep,
        loaded_future: Future[tuple[LoadedLocalBatch, ...] | None],
    ) -> _HostReadyStep:
        loaded = loaded_future.result()
        if not self._topology.is_data_owner:
            self._state.mark_ready(
                reservation.local_batch_offset_start,
                reservation.local_batch_offset_end,
            )
            return _HostReadyStep(None, tuple(None for _ in range(self._planner.micro_batch_num)))
        if loaded is None or self._local_batch_redistributor is None:
            raise ValueError("Online data-owner preparation did not produce local batches and a redistributor.")
        local_metadata = tuple(
            OnlineLocalBatchMetadata(
                local_batch_meta=item.metadata,
                tensor_spec=self._local_batch_redistributor.describe_local_batch(item.data),
            )
            for item in loaded
        )
        global_metadata = self._metadata_synchronizer.gather(
            local_metadata,
            self._topology.data_owner_ranks,
        )
        if any(not isinstance(item, OnlineLocalBatchMetadata) for item in global_metadata):
            raise ValueError("Online metadata synchronization returned invalid local-batch metadata.")
        plan = self._planner.plan(
            tuple(item.local_batch_meta for item in global_metadata),
            step=reservation.step,
            local_batch_offset_start=reservation.local_batch_offset_start,
        )
        batches_by_position = self._local_batch_redistributor.redistribute(
            tuple(item.data for item in loaded),
            plan,
            self._topology,
            global_metadata,
        )
        local_batches = tuple(
            batches_by_position[plan.local_batch_for(self._topology.data_rank, index).source_position]
            for index in range(self._planner.micro_batch_num)
        )
        self._state.mark_ready(
            reservation.local_batch_offset_start,
            reservation.local_batch_offset_end,
        )
        return _HostReadyStep(plan, local_batches)

    def _consume_active_local_batch(self, micro_batch_index: int) -> tuple[BatchPlan, LocalBatch]:
        if self._closed:
            raise ValueError("Cannot consume local batches from a closed DistributedDataset.")
        reservation = self._require_active_reservation()
        if micro_batch_index != self._next_micro_batch_index:
            raise ValueError(
                f"Expected microbatch {self._next_micro_batch_index}, but got {micro_batch_index}."
            )
        if self._double_buffer:
            return self._consume_buffered_local_batch(reservation, micro_batch_index)
        if self._active_future is None:
            self._active_future = self._producer_executor.submit(
                self._prepare_distributed_local_batch,
                reservation,
                micro_batch_index,
            )
        prepared = self._active_future.result()
        self._active_future = None
        self._next_micro_batch_index += 1
        return prepared.plan, prepared.local_batch

    def _schedule_buffered_local_batch(self, reservation: _ReservedStep, micro_batch_index: int) -> None:
        key = (reservation.step, micro_batch_index)
        slot = self._buffer_slots[self._buffer_slot_index(reservation.step, micro_batch_index)]
        if slot.future is not None:
            if slot.key == key:
                return
            raise ValueError(f"Double-buffer slot {slot.index} still holds local batch {slot.key}.")
        slot.key = key
        slot.future = self._producer_executor.submit(
            self._prepare_distributed_local_batch,
            reservation,
            micro_batch_index,
        )

    def _consume_buffered_local_batch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> tuple[BatchPlan, LocalBatch]:
        key = (reservation.step, micro_batch_index)
        slot = self._buffer_slots[self._buffer_slot_index(reservation.step, micro_batch_index)]
        if slot.key != key or slot.future is None:
            raise ValueError(f"Double-buffer slot {slot.index} does not contain expected local batch {key}.")
        try:
            prepared = slot.future.result()
        finally:
            slot.key = None
            slot.future = None
        if prepared.ready_event is not None:
            device_module = _stream_device_module(self._data_stream_device)
            prepared.ready_event.wait(device_module.current_stream(self._data_stream_device))
        self._next_micro_batch_index += 1
        if self._next_micro_batch_index < self._planner.micro_batch_num:
            self._schedule_buffered_local_batch(reservation, self._next_micro_batch_index)
        elif self._prepared_steps:
            self._schedule_buffered_local_batch(self._prepared_steps[0], 0)
        return prepared.plan, prepared.local_batch

    def _prepare_distributed_local_batch(
        self,
        reservation: _ReservedStep,
        micro_batch_index: int,
    ) -> _PreparedLocalBatch:
        if reservation.host_future is None:
            raise ValueError("Reserved step has no Host-prefetch future.")
        stream_context = nullcontext()
        if self._data_stream is not None:
            device_module = _stream_device_module(self._data_stream_device)
            stream_context = device_module.stream(self._data_stream)
        with stream_context:
            host_step = reservation.host_future.result()
            owner_local_batch = host_step.local_batches[micro_batch_index]
            if owner_local_batch is not None and self._prepare_local_batch is not None:
                owner_local_batch = self._prepare_local_batch(owner_local_batch)
            received_plan, rank_data = self._model_parallel_distributor.distribute(
                owner_local_batch,
                host_step.plan,
                self._topology,
            )
            self._validate_received_plan(received_plan, reservation)
            planned = received_plan.local_batch_for(self._topology.data_rank, micro_batch_index)
            local_batch = LocalBatch(
                plan_id=received_plan.plan_id,
                global_rank=self._topology.global_rank,
                data_rank=self._topology.data_rank,
                cp_rank=self._topology.cp_rank,
                micro_batch_index=micro_batch_index,
                local_batch_id=planned.meta.local_batch_id,
                data=rank_data,
            )
            ready_event = None
            if self._data_stream is not None:
                ready_event = device_module.Event()
                ready_event.record(self._data_stream)
        return _PreparedLocalBatch(received_plan, local_batch, ready_event)

    def _validate_received_plan(self, plan: BatchPlan, reservation: _ReservedStep) -> None:
        if plan.step != reservation.step:
            raise ValueError(f"Expected optimizer step {reservation.step}, but received plan step {plan.step}.")
        if plan.data_parallel_size != self._planner.data_parallel_size:
            raise ValueError("Received BatchPlan dimensions do not match the distributed dataset planner.")
        expected_window = (
            reservation.local_batch_offset_start,
            reservation.local_batch_offset_end,
            self._planner.micro_batch_num,
        )
        actual_window = (
            plan.local_batch_offset_start,
            plan.local_batch_offset_end,
            plan.micro_batch_num,
        )
        if actual_window != expected_window:
            raise ValueError(f"Received BatchPlan window {actual_window} does not match expected {expected_window}.")
        if reservation.plan_id is None:
            reservation.plan_id = plan.plan_id
        elif plan.plan_id != reservation.plan_id:
            raise ValueError("Local batches received inconsistent whole-step plan IDs.")

    def _complete_active_step(self, step: DistributedDataStep, plan_id: str) -> None:
        if step is not self._active_step:
            raise ValueError("Completed DistributedDataStep is not the dataset's active step.")
        reservation = self._require_active_reservation()
        if self._next_micro_batch_index != self._planner.micro_batch_num:
            raise ValueError("Cannot complete a DistributedDataStep before every local batch is produced.")
        self._state.mark_delivered(
            plan_id,
            reservation.local_batch_offset_start,
            reservation.local_batch_offset_end,
        )
        self._fill_prefetch()

    def _require_active_reservation(self) -> _ReservedStep:
        if self._active_reservation is None:
            raise ValueError("DistributedDataset has no active optimizer-step reservation.")
        return self._active_reservation

    def _buffer_slot_index(self, step: int, micro_batch_index: int) -> int:
        sequence = step * self._planner.micro_batch_num + micro_batch_index
        return sequence % len(self._buffer_slots)
