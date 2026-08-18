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
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from hyper_parallel.distributed_data.distributor import (
    MetadataSynchronizer,
    OwnerPayloadRedistributor,
    PayloadDistributor,
)
from hyper_parallel.distributed_data.materializer import (
    LoadedSample,
    MetadataSource,
    OnlineSampleSource,
    RankMaterializer,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import BatchPlan, DistributedDataStep, RankPayload
from hyper_parallel.distributed_data.state import DatasetStateTracker
from hyper_parallel.distributed_data.topology import DataTopology


@dataclass(frozen=True)
class _PreparedOwnerStep:
    plan: BatchPlan
    micro_batches: tuple[Any, ...]


@dataclass(frozen=True)
class _PreparedOnlineWindow:
    step: int
    cursor_start: int
    samples: tuple[LoadedSample, ...]


class DistributedDataset(Iterator[DistributedDataStep]):
    """Plan globally, materialize on data owners, and distribute per step.

    The iterator yields one :class:`DistributedDataStep`, containing all
    local microbatches for one optimizer step. Call :meth:`commit` only after
    that optimizer step succeeds.
    """

    def __init__(
        self,
        *,
        topology: DataTopology,
        metadata_source: MetadataSource | None,
        planner: DistributedBatchPlanner,
        rank_materializer: RankMaterializer,
        metadata_synchronizer: MetadataSynchronizer,
        payload_distributor: PayloadDistributor,
        prefetch_steps: int = 2,
        prepare_payload: Callable[[Any], Any] | None = None,
        online_sample_source: OnlineSampleSource | None = None,
        owner_payload_redistributor: OwnerPayloadRedistributor | None = None,
    ) -> None:
        """Initialize planning, materialization, communication, and prefetch components."""
        if planner.data_world_size != topology.data_world_size:
            raise ValueError(
                f"Planner data_world_size={planner.data_world_size} does not match "
                f"topology data_world_size={topology.data_world_size}."
            )
        if (metadata_source is None) == (online_sample_source is None):
            raise ValueError("Configure exactly one of metadata_source and online_sample_source.")
        if online_sample_source is not None and owner_payload_redistributor is None:
            raise ValueError("Online metadata requires an owner_payload_redistributor.")
        self._topology = topology
        self._metadata_source = metadata_source
        self._online_sample_source = online_sample_source
        self._planner = planner
        self._rank_materializer = rank_materializer
        self._metadata_synchronizer = metadata_synchronizer
        self._owner_payload_redistributor = owner_payload_redistributor
        self._payload_distributor = payload_distributor
        self._prepare_payload = prepare_payload
        source_size = len(metadata_source) if metadata_source is not None else len(online_sample_source)
        self._source_size = source_size
        self._state = DatasetStateTracker(
            source_size,
            planner.local_samples_per_step,
            prefetch_steps,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hp-data-prefetch")
        self._prepared_steps: deque[Future[_PreparedOwnerStep | _PreparedOnlineWindow] | None] = deque()
        self._closed = False

    def __len__(self) -> int:
        """Return complete optimizer steps in this rank's candidate shard."""
        return self._source_size // self._planner.local_samples_per_step

    def __iter__(self) -> "DistributedDataset":
        """Return this stateful optimizer-step iterator."""
        return self

    def __next__(self) -> DistributedDataStep:
        """Return the next planned optimizer step."""
        if self._closed:
            raise StopIteration
        if self._state.has_delivered_unconsumed:
            raise ValueError("Commit the current DistributedDataStep before requesting the next step.")
        self._fill_prefetch()
        if not self._prepared_steps:
            raise StopIteration
        prepared_future = self._prepared_steps.popleft()
        prepared = prepared_future.result() if prepared_future is not None else None
        if isinstance(prepared, _PreparedOnlineWindow):
            prepared = self._plan_online_window(prepared)
        step = self._distribute_step(prepared)
        self._state.mark_delivered(step.plan)
        return step

    @property
    def consumed_offset(self) -> int:
        """Return the offset after the last successful optimizer step."""
        return self._state.consumed_offset

    @property
    def prepared_offset(self) -> int:
        """Return the offset reserved for bounded data preparation."""
        return self._state.prepared_offset

    def commit(self, replay_id: str) -> None:
        """Commit one delivered plan after optimizer-step success."""
        self._state.commit(replay_id)
        self._fill_prefetch()

    def state_dict(self) -> dict[str, int]:
        """Return checkpoint state at the successfully consumed offset."""
        return self._state.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the consumed offset before iteration starts."""
        if self._prepared_steps:
            raise ValueError("Cannot restore distributed dataset state after data preparation has started.")
        self._state.load_state_dict(state_dict)

    def close(self) -> None:
        """Wait for current prefetch work and close the producer thread."""
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._prepared_steps.clear()
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
            if self._topology.is_data_owner:
                if self._metadata_source is not None:
                    local_metadata = tuple(
                        self._metadata_source.get(index)
                        for index in range(cursor_start, cursor_end)
                    )
                    candidates = self._metadata_synchronizer.gather(local_metadata, self._topology.owner_ranks)
                    plan = self._planner.plan(candidates, step=step, cursor_start=cursor_start)
                    self._prepared_steps.append(self._executor.submit(self._materialize_plan, plan))
                else:
                    self._prepared_steps.append(
                        self._executor.submit(self._load_online_window, step, cursor_start, cursor_end)
                    )
            else:
                self._prepared_steps.append(None)

    def _load_online_window(self, step: int, cursor_start: int, cursor_end: int) -> _PreparedOnlineWindow:
        if self._online_sample_source is None:
            raise ValueError("Online sample source is not configured.")
        samples = tuple(self._online_sample_source.get(index) for index in range(cursor_start, cursor_end))
        return _PreparedOnlineWindow(step, cursor_start, samples)

    def _plan_online_window(self, window: _PreparedOnlineWindow) -> _PreparedOwnerStep:
        if self._owner_payload_redistributor is None:
            raise ValueError("Online owner payload redistribution is not configured.")
        local_metadata = tuple(sample.metadata for sample in window.samples)
        candidates = self._metadata_synchronizer.gather(local_metadata, self._topology.owner_ranks)
        plan = self._planner.plan(candidates, step=window.step, cursor_start=window.cursor_start)
        payload_by_position = self._owner_payload_redistributor.redistribute(
            tuple(sample.payload for sample in window.samples),
            plan,
            self._topology,
        )
        micro_batches = []
        for micro_batch_index in range(self._planner.micro_batch_count):
            planned_samples = plan.samples_for(self._topology.data_rank, micro_batch_index)
            samples = [payload_by_position[sample.source_position] for sample in planned_samples]
            micro_batches.append(self._rank_materializer.collate(samples))
        return _PreparedOwnerStep(plan, tuple(micro_batches))

    def _materialize_plan(self, plan: BatchPlan) -> _PreparedOwnerStep:
        micro_batches = []
        for micro_batch_index in range(self._planner.micro_batch_count):
            owner_payload = self._rank_materializer.materialize(
                plan,
                self._topology.data_rank,
                micro_batch_index,
            )
            micro_batches.append(owner_payload)
        return _PreparedOwnerStep(plan, tuple(micro_batches))

    def _distribute_step(self, prepared: _PreparedOwnerStep | None) -> DistributedDataStep:
        plan = prepared.plan if prepared is not None else None
        payloads = []
        received_plan = plan
        for micro_batch_index in range(self._planner.micro_batch_count):
            owner_payload = prepared.micro_batches[micro_batch_index] if prepared is not None else None
            if owner_payload is not None and self._prepare_payload is not None:
                owner_payload = self._prepare_payload(owner_payload)
            current_plan, rank_data = self._payload_distributor.distribute(
                owner_payload,
                plan,
                self._topology,
            )
            if received_plan is not None and received_plan.replay_id != current_plan.replay_id:
                raise ValueError("Payload microbatches received inconsistent BatchPlan replay IDs.")
            received_plan = current_plan
            planned_samples = current_plan.samples_for(self._topology.data_rank, micro_batch_index)
            payloads.append(
                RankPayload(
                    replay_id=current_plan.replay_id,
                    global_rank=self._topology.global_rank,
                    data_rank=self._topology.data_rank,
                    cp_rank=self._topology.cp_rank,
                    micro_batch_index=micro_batch_index,
                    sample_ids=tuple(sample.meta.sample_id for sample in planned_samples),
                    data=rank_data,
                )
            )

        if received_plan is None:
            raise ValueError("Payload distributor did not provide a BatchPlan.")
        return DistributedDataStep(received_plan, tuple(payloads))
