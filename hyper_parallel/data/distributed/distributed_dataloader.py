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
"""Dataset Reader, Planner, and Data Constructor orchestration."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import torch

from hyper_parallel.data.parallel.batch_sampler import (
    sampler_state_for_load,
)
from hyper_parallel.data.distributed.balance_logging import logger as balance_logger
from hyper_parallel.data.distributed.data_constructor import PackingDataConstructor
from hyper_parallel.data.distributed.batch_sampler import BatchSamplerReader
from hyper_parallel.data.distributed.device_prefetch import DevicePrefetchedStep, DeviceStepPrefetcher, _record_stream
from hyper_parallel.data.distributed.planner import DynamicPackingPlanner
from hyper_parallel.data.distributed.metadata import PlannedSampleLoader
from hyper_parallel.data.distributed.topology import DataTopology
from hyper_parallel.data.distributed.native_source import NativeWindowSource, PreparedWindow as _PreparedWindow
from hyper_parallel.data.distributed.process_data import DataRequest, create_data_worker
from hyper_parallel.data.distributed.transport import (
    DataGroups,
    DataPlaneTransport,
    ModelParallelTransport,
)


def _validate_loader_components(
        *,
        topology: DataTopology,
        dataset_reader_ranks: tuple[int, ...],
        dataset_reader: Any | None,
        metadata_reader: BatchSamplerReader | None,
        direct_sample_loader: PlannedSampleLoader | None,
        metadata_mode: bool,
) -> None:
    """Check Reader ownership and require mode-compatible loading components."""
    if not isinstance(metadata_mode, bool):
        raise ValueError("metadata_mode must be boolean.")
    if metadata_mode and dataset_reader is not None:
        raise ValueError("Metadata mode must not configure an online Dataset Reader.")
    if not metadata_mode and (metadata_reader is not None or direct_sample_loader is not None):
        raise ValueError("Online mode must not configure metadata loading components.")
    is_reader = topology.global_rank in dataset_reader_ranks
    planning_reader = metadata_reader if metadata_mode else dataset_reader
    if is_reader != (planning_reader is not None):
        raise ValueError("Dataset Reader ownership does not match dataset_reader_ranks.")
    if not metadata_mode and planning_reader is not None and not callable(
            getattr(planning_reader, "prepare_next_step", None)
    ):
        raise ValueError("Online Dataset Reader must provide prepare_next_step().")
    _validate_metadata_loader_owner(
        topology,
        metadata_mode=metadata_mode,
        direct_sample_loader=direct_sample_loader,
    )


def _validate_metadata_loader_owner(
        topology: DataTopology,
        *,
        metadata_mode: bool,
        direct_sample_loader: PlannedSampleLoader | None,
) -> None:
    if not metadata_mode:
        return
    if topology.is_constructor != (direct_sample_loader is not None):
        raise ValueError("Every metadata Data Constructor must own one plan-aware sample loader.")


@dataclass
class _PreparedBatch:
    """Carry one batch and its delivery state through parent-side staging."""

    batch: Any
    checkpoint: dict[str, Any] | None = None
    balance_logs: list[str] = field(default_factory=list)
    ready_event: Any = None
    error: BaseException | None = None


class DistributedDataLoader(Iterator[Any]):
    """Prepare balancing windows and deliver one local micro-batch at a time.

    The configured process owns reads, planning, construction and data collectives.
    The training process stages H2D and broadcasts only at delivery.
    Host windows and device inputs have independent buffering limits.
    Checkpoints advance at complete optimizer-step delivery boundaries.
    """

    def __init__(
            self,
            *,
            topology: DataTopology,
            dataset_reader_ranks: tuple[int, ...],
            dataset_reader: Any | None,
            metadata_reader: BatchSamplerReader | None,
            direct_sample_loader: PlannedSampleLoader | None,
            metadata_mode: bool,
            planner: DynamicPackingPlanner,
            data_constructor: PackingDataConstructor,
            data_plane: DataPlaneTransport,
            model_transport: ModelParallelTransport,
            device_prefetch: DeviceStepPrefetcher | None = None,
            initial_epoch: int = 0,
            initial_consumed_samples: int = 0,
            broadcast_plan: bool = True,
            log_balance: bool = False,
            accumulation_steps: int = 1,
            balance_window_size: int = 1,
            device_prefetch_depth: int = 1,
            use_data_process: bool = True,
            device: Any = "cpu",
            groups: DataGroups | None = None,
    ) -> None:
        """Store the fully validated runtime components."""
        _validate_loader_components(
            topology=topology,
            dataset_reader_ranks=dataset_reader_ranks,
            dataset_reader=dataset_reader,
            metadata_reader=metadata_reader,
            direct_sample_loader=direct_sample_loader,
            metadata_mode=metadata_mode,
        )
        for name, value in (("accumulation_steps", accumulation_steps), ("balance_window_size", balance_window_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (isinstance(device_prefetch_depth, bool) or not isinstance(device_prefetch_depth, int)
                or device_prefetch_depth < 0):
            raise ValueError("device_prefetch_depth must be a non-negative integer.")
        if not isinstance(use_data_process, bool):
            raise ValueError("use_data_process must be boolean.")
        self._source = NativeWindowSource(
            topology=topology, dataset_reader_ranks=dataset_reader_ranks, dataset_reader=dataset_reader,
            metadata_reader=metadata_reader, direct_sample_loader=direct_sample_loader, metadata_mode=metadata_mode,
            planner=planner, data_constructor=data_constructor, initial_epoch=initial_epoch,
            initial_consumed_samples=initial_consumed_samples, broadcast_plan=broadcast_plan,
            log_balance=log_balance, accumulation_steps=accumulation_steps, balance_window_size=balance_window_size,
        )
        self._topology = topology
        self._data_plane = data_plane
        self._groups = groups
        self._model_transport = model_transport
        self._device_prefetch = device_prefetch
        self._worker = None
        self._device_prefetch_depth = device_prefetch_depth
        self._use_data_process = use_data_process
        self._device = torch.device(device)
        self._current: _PreparedWindow | None = None
        self._host_request: DataRequest | None = None
        self._device_batches: deque[_PreparedBatch] = deque()
        self._prefetch_target = 0
        self._consumer_stream: Any = None
        # Each staged batch spends one input slot; advancing returns it with the
        # previous consumer's completion event, even if that work is still running.
        # Include the active batch so device lookahead can overlap its computation.
        self._prefetch_slots: deque[Any] = deque([None] * (device_prefetch_depth + 1))
        self._device_stopped = False
        self._batch_error: BaseException | None = None
        self._last_ready_event: Any = None
        self._has_delivered = False
        self._closed = False
        self._delivered_state = self._source.snapshot()

    def __iter__(self) -> DistributedDataLoader:
        """Return the rank-local iterator."""
        return self

    def __next__(self) -> Any:
        """Deliver one batch and refill the configured device lookahead."""
        if self._closed or self._delivered_state["stopped"]:
            raise StopIteration
        if self._batch_error is not None:
            raise self._batch_error
        self._record_consumption()
        if self._device.type in ("npu", "cuda"):
            prepared = self._next_device_batch()
        else:
            try:
                prepared = self._next_host_batch()
            except BaseException as error:
                self._model_transport.broadcast(None, error=error)
                raise
            prepared.batch = self._broadcast_batch(prepared.batch)
        if prepared.batch is None:
            self._delivered_state["stopped"] = True
            raise StopIteration
        batch = prepared.batch
        if prepared.ready_event is not None:
            module = getattr(torch, self._device.type)
            self._consumer_stream = module.current_stream(self._device)
            _record_stream(batch, self._consumer_stream)
        if prepared.checkpoint is not None:
            self._delivered_state = prepared.checkpoint
        for message in prepared.balance_logs:
            balance_logger.info("%s", message)
        if (not self._has_delivered and self._device_prefetch_depth and self._topology.is_constructor
                and self._device.type in ("npu", "cuda")):
            self._prefetch_device_batches(1, blocking=True)
        self._has_delivered = True
        self.prefetch()
        return batch

    def _record_consumption(self) -> None:
        # Advancing the iterator retires the previous batch. Callers must finish
        # submitting its uses and release input references before advancing.
        if self._consumer_stream is not None:
            prefetch_wait_event = getattr(torch, self._device.type).Event()
            prefetch_wait_event.record(self._consumer_stream)
            self._prefetch_slots.append(prefetch_wait_event)
            self._consumer_stream = None

    def _next_host_batch(self) -> _PreparedBatch:
        """Pop one Host batch and its delivery state, prefetching the following window."""
        if self._current is None or not self._current.batches:
            if self._current is not None and self._current.exhausted:
                return _PreparedBatch(None)
            self._request_host_window()
            self._current = self._host_request.result()
            self._host_request = None
            if not self._current.batches:
                return _PreparedBatch(None)
            if self._use_data_process:
                self._request_host_window()
        batch = self._current.batches.popleft()
        logs, self._current.balance_logs = self._current.balance_logs, []
        checkpoint = self._current.checkpoint if not self._current.batches else None
        return _PreparedBatch(batch, checkpoint, logs)

    def prefetch(self) -> None:
        """Prefetch Constructor H2D without any model-peer communication."""
        if self._closed or self._delivered_state["stopped"] or self._batch_error is not None:
            return
        if self._use_data_process:
            self._request_host_window()
        if self._device.type in ("npu", "cuda") and self._topology.is_constructor:
            self._prefetch_target = self._device_prefetch_depth
            self._prefetch_device_batches(self._prefetch_target)

    def _host_batch_ready(self) -> bool:
        if self._current is not None and (self._current.batches or self._current.exhausted):
            return True
        return self._host_request is not None and self._host_request.is_ready()

    def _prefetch_device_batches(self, count: int, *, blocking: bool = False, on_demand: bool = False) -> None:
        """Fill available slots; only warmup, demand and explicit drains can wait."""
        while len(self._device_batches) < count and self._prefetch_slots and not self._device_stopped:
            # Only delivery is collective across data ranks. A Constructor may
            # stage buffered inputs alone, but cannot open a synchronous window alone.
            if not self._use_data_process and not on_demand and not (self._current and self._current.batches):
                break
            event = self._prefetch_slots[0]
            try:
                if not blocking and (not self._host_batch_ready() or (event is not None and not event.query())):
                    break
                prepared = self._stage_device_batch(event if blocking else None)
            except BaseException as error:  # pylint: disable=broad-exception-caught
                # A failed speculative copy or read belongs after earlier queued batches.
                prepared = _PreparedBatch(None, error=error)
            self._prefetch_slots.popleft()
            self._device_batches.append(prepared)
            self._device_stopped = prepared.batch is None

    def _request_host_window(self) -> None:
        """Request one Host window from the configured source owner."""
        if self._host_request is not None or (self._current is not None and self._current.exhausted):
            return
        if self._worker is None:
            if self._device.type in ("npu", "cuda"):
                getattr(torch, self._device.type).set_device(self._device)
            if self._use_data_process:
                self._worker = create_data_worker(self._source, self._data_plane)
            else:
                self._source.bind_transport(self._data_plane)
                self._worker = self._source
        self._host_request = DataRequest(self._worker)

    def _stage_device_batch(self, event: Any) -> _PreparedBatch:
        """Retire a previous input before allocating its replacement on the copy stream."""
        prepared = self._next_host_batch()
        if event is not None:
            event.synchronize()
        if prepared.batch is not None and self._device_prefetch is not None:
            prepared.batch = self._device_prefetch(prepared.batch)
        if isinstance(prepared.batch, DevicePrefetchedStep):
            prepared.ready_event = prepared.batch.ready_event
        return prepared

    def _next_device_batch(self) -> _PreparedBatch:
        """Broadcast on the training stream after waiting for only the current copy."""
        try:
            if self._topology.is_constructor:
                self._prefetch_device_batches(1, blocking=True, on_demand=True)
                prepared = self._device_batches.popleft()
            else:
                prepared = self._next_host_batch()
                prefetch_wait_event = self._prefetch_slots.popleft()
                if prefetch_wait_event is not None:
                    prefetch_wait_event.synchronize()
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            # All preparation failures must reach peers through the broadcast error header.
            prepared = _PreparedBatch(None, error=exc)
        module = getattr(torch, self._device.type)
        consumer_stream = module.current_stream(self._device)
        try:
            batch = prepared.batch
            if isinstance(batch, DevicePrefetchedStep):
                batch = batch.take_batch()
            prepared.batch = self._broadcast_batch(batch) if prepared.error is None else (
                self._model_transport.broadcast(None, error=prepared.error)
            )
        except BaseException as exc:
            self._batch_error = exc
            self._device_stopped = True
            self._device_batches.clear()
            raise
        finally:
            # Explicit drains must cover partially submitted broadcasts on errors
            # without synchronizing the caller's later training work.
            prepared.ready_event = module.Event()
            prepared.ready_event.record(consumer_stream)
            self._last_ready_event = prepared.ready_event
        return prepared

    def _broadcast_batch(self, constructed_batch: Any) -> Any:
        """Broadcast only the batch being delivered, on the calling training thread."""
        return self._model_transport.broadcast(constructed_batch)

    def _fill_requested_device_prefetch(self) -> None:
        """Complete requested Constructor lookahead before waiting or closing."""
        # Only accelerator Constructor staging sets a positive target.
        if (not self._prefetch_target or self._closed or self._delivered_state["stopped"]
                or self._batch_error is not None):
            return
        self._prefetch_device_batches(self._prefetch_target, blocking=True)

    def wait_for_prefetch(self) -> None:
        """Drain buffered window communication before checkpoint teardown or shutdown."""
        self._fill_requested_device_prefetch()
        if self._last_ready_event is not None:
            self._last_ready_event.synchronize()
        ready_event = None
        for prepared in self._device_batches:
            if prepared.ready_event is not None:
                ready_event = prepared.ready_event
        # Queued events belong to the copy stream, independently of the last
        # delivered batch's training-stream marker. Neither drain broadcasts ahead.
        if ready_event is not None:
            ready_event.synchronize()
        if self._host_request is not None:
            self._host_request.result()

    def close(self) -> None:
        """Drain the producer before releasing buffered inputs and reader resources."""
        try:
            with ExitStack() as cleanup:
                if self._groups is not None:
                    cleanup.callback(self._groups.close)
                if self._closed:
                    return
                cleanup.callback(self._source.close)
                if self._worker is not None and self._worker is not self._source:
                    cleanup.callback(self._worker.close)
                if self._device_prefetch is not None:
                    cleanup.callback(self._device_prefetch.close)
                # Ranks can stage different amounts during next(). Catch up to the same
                # lookahead before any child exits its outstanding data collectives.
                self._fill_requested_device_prefetch()
                if self._last_ready_event is not None:
                    self._last_ready_event.synchronize()
        finally:
            self._current = None
            self._host_request = None
            self._device_batches.clear()
            self._consumer_stream = None
            self._prefetch_slots.clear()
            self._batch_error = None
            self._last_ready_event = None
            self._closed = True
            self._worker = None

    def __enter__(self) -> "DistributedDataLoader":
        """Support deterministic cleanup on early loop exit."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release prefetch resources before process-group teardown."""
        self.close()

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the last fully delivered optimizer step without waiting.

        Save training checkpoints only after the corresponding optimizer update.
        An incomplete step leaves this cursor at that step's beginning.
        """
        state = self._delivered_state
        if "_index_sampler_state" not in state:
            # Old non-Reader checkpoints may lack an absolute source cursor.
            # Keep their same-mode progress without inventing a portable position.
            return copy.deepcopy(dict(state, global_rank=self._topology.global_rank))
        return copy.deepcopy({
            "_index_sampler_state": state["_index_sampler_state"],
            "_iterator_finished": state["stopped"],
            "dataset_state": state["dataset_state"],
        })

    def state_dict_for_load(self) -> dict[str, Any]:
        """Request the shared cursor and legacy Reader aliases without defaulting to zero."""
        return sampler_state_for_load(self.state_dict())

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the native cursor before starting the data process.

        Args:
            state_dict: Saved native cursor at a complete optimizer-step boundary.
        """
        self._validate_restore_boundary()
        self._source.load_state_dict(state_dict)
        self._delivered_state = self._source.snapshot()

    def _validate_restore_boundary(self) -> None:
        if self._last_ready_event is not None or self._worker is not None or self._has_delivered:
            raise ValueError("load_state_dict must run before distributed iteration starts.")

    def set_epoch(self, epoch: int) -> None:
        """Reset the native source while retaining the data worker and copy stream.

        Args:
            epoch: Non-negative Dataset epoch.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if self._closed:
            raise ValueError("set_epoch cannot restart a closed loader.")
        if (self._has_delivered or self._delivered_state["step"] != 0) and not self._delivered_state["stopped"]:
            raise ValueError("set_epoch requires a fresh or exhausted optimizer-step loader.")
        if epoch == self._delivered_state["epoch"] and not self._delivered_state["stopped"]:
            return
        self.wait_for_prefetch()
        if self._device_prefetch is not None:
            self._device_prefetch.close()
        self._source.set_epoch(epoch)
        if self._worker is not None and self._worker is not self._source:
            self._worker.set_epoch(epoch)
        self._current = None
        self._host_request = None
        self._device_batches.clear()
        self._consumer_stream = None
        self._prefetch_slots = deque([None] * (self._device_prefetch_depth + 1))
        self._device_stopped = False
        self._prefetch_target = 0
        self._batch_error = None
        self._last_ready_event = None
        self._has_delivered = False
        self._delivered_state = self._source.snapshot()


__all__ = ["DistributedDataLoader"]
