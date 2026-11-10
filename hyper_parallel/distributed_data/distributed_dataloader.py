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
import logging
import math
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Any, Union

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.parallel.batch_sampler import (
    get_sampler_checkpoint_state, has_torchdata_state, sampler_optimizer_step_progress, sampler_state_for_load,
)
from hyper_parallel.distributed_data.balance_logging import format_balance_stats, logger as balance_logger
from hyper_parallel.distributed_data.data_constructor import PackingDataConstructor
from hyper_parallel.distributed_data.batch_sampler import BatchSamplerReader, _ReaderPlanInput
from hyper_parallel.distributed_data.device_prefetch import DevicePrefetchedStep, DeviceStepPrefetcher, _record_stream
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    DistributedPackingPlan,
    SampleKey,
    SampleMetadata,
)
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    PreparedPayloadExchange,
)


_PlanControl = Union[DistributedPackingPlan, tuple[DistributedPackingPlan, ...], None]


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
class _PreparedWindow:
    """Retain one Host balancing window and an optional complete-step checkpoint."""

    batches: deque[Any]
    exhausted: bool
    checkpoint: dict[str, Any] | None
    balance_logs: list[str] = field(default_factory=list)


@dataclass
class _PreparedBatch:
    """Carry one batch and its delivery state through caller-side staging."""

    batch: Any
    checkpoint: dict[str, Any] | None = None
    balance_logs: list[str] = field(default_factory=list)
    ready_event: Any = None
    error: BaseException | None = None


class DistributedDataLoader(Iterator[Any]):
    """Prepare balancing windows and deliver one local micro-batch at a time.

    One Host producer owns reads, planning, data-plane collectives and pinning.
    The caller prefetches H2D and broadcasts only the batch being delivered.
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
            device: Any = "cpu",
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
        self._topology = topology
        self._dataset_reader_ranks = frozenset(dataset_reader_ranks)
        self._dataset_reader = dataset_reader
        self._metadata_reader = metadata_reader
        self._direct_sample_loader = direct_sample_loader
        self._metadata_mode = metadata_mode
        self._broadcast_plan = broadcast_plan
        self._log_balance = log_balance
        self._planner = planner
        self._data_constructor = data_constructor
        self._data_plane = data_plane
        self._model_transport = model_transport
        self._device_prefetch = device_prefetch
        self._epoch = initial_epoch
        self._step = 0
        self._initial_consumed_samples: int | None = initial_consumed_samples
        self._stopped = False
        self._pending_local_keys: set[SampleKey] = set()
        self._window_plans: deque[DistributedPackingPlan] = deque()
        for name, value in (("accumulation_steps", accumulation_steps), ("balance_window_size", balance_window_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self._accumulation_steps = accumulation_steps
        self._balance_window_size = balance_window_size
        if (isinstance(device_prefetch_depth, bool) or not isinstance(device_prefetch_depth, int)
                or device_prefetch_depth < 0):
            raise ValueError("device_prefetch_depth must be a non-negative integer.")
        self._device_prefetch_depth = device_prefetch_depth
        self._device = torch.device(device)
        self._current: _PreparedWindow | None = None
        self._future: Future[_PreparedWindow] | None = None
        self._requests: Queue[Future[_PreparedWindow] | None] = Queue(maxsize=1)
        self._prefetch_thread: Thread | None = None
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
        self._balance_logs: list[str] = []
        self._delivered_state = self._snapshot_state()

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
            prepared = self._next_host_batch()
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
            self._prefetch_host_window()
            self._current = self._future.result()
            self._future = None
            if not self._current.batches:
                return _PreparedBatch(None)
            self._prefetch_host_window()
        batch = self._current.batches.popleft()
        logs, self._current.balance_logs = self._current.balance_logs, []
        checkpoint = self._current.checkpoint if not self._current.batches else None
        return _PreparedBatch(batch, checkpoint, logs)

    def prefetch(self) -> None:
        """Prefetch Constructor H2D without any model-peer communication."""
        if self._closed or self._delivered_state["stopped"] or self._batch_error is not None:
            return
        if (self._device.type not in ("npu", "cuda") or self._device_prefetch_depth == 0
                or not self._topology.is_constructor):
            self._prefetch_target = 0
            self._prefetch_host_window()
            return
        self._prefetch_device_batches(self._device_prefetch_depth)

    def _host_batch_ready(self) -> bool:
        """Check Host readiness without waiting for speculative window work."""
        if self._current is not None and (self._current.batches or self._current.exhausted):
            return True
        self._prefetch_host_window()
        return self._future.done()

    def _prefetch_device_batches(self, count: int, *, blocking: bool = False) -> None:
        """Fill an active Constructor's H2D queue after caller-side lifecycle checks."""
        self._prefetch_target = count
        while len(self._device_batches) < count and self._prefetch_slots and not self._device_stopped:
            prefetch_wait_event = self._prefetch_slots[0]
            if not blocking and (
                not self._host_batch_ready()
                or (prefetch_wait_event is not None and not prefetch_wait_event.query())
            ):
                return
            try:
                prepared = self._stage_device_batch(prefetch_wait_event)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                # Preserve FIFO delivery: a speculative failure must not prevent
                # earlier batches from training or enter a peer collective early.
                prepared = _PreparedBatch(None, error=exc)
                self._device_stopped = True
            self._prefetch_slots.popleft()
            self._device_batches.append(prepared)

    def _prefetch_host_window(self) -> None:
        """Submit one Host window for an active loader, preserving double buffering."""
        if self._future is not None:
            return
        if self._current is not None and self._current.exhausted:
            return
        self._future = Future()
        if self._prefetch_thread is None:
            self._prefetch_thread = Thread(
                target=self._run_prefetch, name=f"hp-data-prefetch-rank-{self._topology.global_rank}", daemon=True,
            )
            self._prefetch_thread.start()
        self._requests.put(self._future)

    def _stage_device_batch(self, prefetch_wait_event: Any) -> _PreparedBatch:
        """Keep copy events local until the corresponding batch is consumed."""
        prepared = self._next_host_batch()
        if prefetch_wait_event is not None:
            prefetch_wait_event.synchronize()
        prepared.batch = self._stage_batch(prepared.batch)
        if isinstance(prepared.batch, DevicePrefetchedStep):
            prepared.ready_event = prepared.batch.ready_event
        self._device_stopped = prepared.batch is None
        return prepared

    def _next_device_batch(self) -> _PreparedBatch:
        """Broadcast on the training stream after waiting for only the current copy."""
        try:
            if self._topology.is_constructor:
                self._prefetch_device_batches(1, blocking=True)
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
            self._device_batches.clear()
            raise
        finally:
            # Explicit drains must cover partially submitted broadcasts on errors
            # without synchronizing the caller's later training work.
            prepared.ready_event = module.Event()
            prepared.ready_event.record(consumer_stream)
            self._last_ready_event = prepared.ready_event
        return prepared

    def _run_prefetch(self) -> None:
        """Serve window requests on a persistent producer and propagate failures via futures."""
        stream = None
        try:
            while True:
                future = self._requests.get()
                if future is None:
                    return
                try:
                    if stream is None and self._device.type in ("npu", "cuda"):
                        module = getattr(torch, self._device.type)
                        module.set_device(self._device)
                        stream = module.Stream(device=self._device)
                    prefetch_stream_context = (
                        getattr(torch, self._device.type).stream(stream) if stream is not None else nullcontext()
                    )
                    with prefetch_stream_context:
                        window = self._prepare_window()
                except BaseException as exc:  # pylint: disable=broad-exception-caught
                    # A worker interrupt must complete its future instead of stranding result().
                    future.set_exception(exc)
                else:
                    future.set_result(window)
                    del window
                # An idle producer must not retain a discarded epoch's inputs.
                del future
        finally:
            if stream is not None:
                stream.synchronize()

    def _prepare_window(self) -> _PreparedWindow:
        """Prepare Host batches up to the window, optimizer-step or epoch boundary."""
        window_micro_batch_num = min(
            self._balance_window_size, self._accumulation_steps - self._step % self._accumulation_steps,
        )
        self._balance_logs = []
        self._prepare_local_metadata(window_micro_batch_num)
        batches = deque()
        exhausted = False
        for _ in range(window_micro_batch_num):
            try:
                batch = self._prepare_batch()
            except StopIteration:
                exhausted = True
                break
            batches.append(batch)
        # Intermediate windows must replay from the optimizer step's beginning.
        checkpoint = self._snapshot_state() if not exhausted and self._step % self._accumulation_steps == 0 else None
        return _PreparedWindow(batches, exhausted, checkpoint, balance_logs=self._balance_logs)

    def _prepare_batch(self) -> Any:
        """Construct one Host round; accelerator peers receive it at device delivery."""
        if self._stopped:
            raise StopIteration
        batch = self._prepare_next_batch()
        if batch is not None and self._device_prefetch is not None:
            batch = self._device_prefetch.prepare_host(batch)
        if self._device.type == "cpu":
            batch = self._broadcast_batch(self._stage_batch(batch))
            self._stopped = batch is None
        if self._stopped:
            raise StopIteration
        return self._consume_batch(batch)

    def _stage_batch(self, batch: Any) -> Any:
        if batch is not None and self._device_prefetch is not None:
            return self._device_prefetch.stage_prepared(batch)
        return batch

    def _prepare_next_batch(self) -> Any:
        """Select, balance, route, and construct samples without model-group broadcast."""
        if self._data_plane.is_member:
            return self._produce_on_data_plane()
        return None

    def _broadcast_batch(self, constructed_batch: Any) -> Any:
        """Broadcast one batch on the caller stream, or on the Host producer for CPU."""
        return self._model_transport.broadcast(constructed_batch)

    def _consume_batch(self, received: Any) -> Any:
        """Commit Host preparation; delivery publishes checkpoints only after broadcast."""
        # Non-Constructors hold placeholders; the caller receives both
        # real batches and the EOF sentinel from their Constructor's broadcast.
        planning_reader = self._planning_reader()
        if planning_reader is not None:
            planning_reader.commit(self._pending_local_keys)
        self._pending_local_keys.clear()
        self._step += 1
        return received

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
        if self._future is not None:
            self._future.result()

    def close(self) -> None:
        """Drain the producer before releasing buffered inputs and reader resources."""
        if self._closed:
            return
        # Constructors may have skipped optional staging at different times.
        # Complete the requested lookahead before any producer leaves its collectives;
        # discarded input errors still belong to next()/wait_for_prefetch(), not close().
        self._fill_requested_device_prefetch()
        if self._last_ready_event is not None:
            self._last_ready_event.synchronize()
        if self._prefetch_thread is not None:
            if self._prefetch_thread.is_alive():
                self._requests.put(None)
            self._prefetch_thread.join()
            self._prefetch_thread = None
        try:
            if self._device_prefetch is not None:
                self._device_prefetch.close()
        finally:
            reader = self._planning_reader()
            if reader is not None:
                reader.close()
            self._current = None
            self._future = None
            self._device_batches.clear()
            self._consumer_stream = None
            self._prefetch_slots.clear()
            self._batch_error = None
            self._last_ready_event = None
            self._balance_logs = []
            self._closed = True

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

    def _snapshot_state(self) -> dict[str, Any]:
        """Capture producer progress at an optimizer-step boundary."""
        reader = self._planning_reader()
        reader_state = reader.state_dict() if reader is not None else None
        payload_state = (
            self._direct_sample_loader.state_dict() if self._direct_sample_loader is not None
            else (reader_state or {}).get("sample_loader")
        )
        state = {
            "epoch": self._epoch,
            "step": self._step,
            "stopped": self._stopped,
            "dataset_state": (
                {key: payload_state[key] for key in ("dataset_size", "seed")} if payload_state is not None else None
            ),
        }
        if reader_state is not None:
            state["_index_sampler_state"] = reader_state["sampler"]
        elif self._initial_consumed_samples is not None:
            # Model peers have no Reader; advance their cursor from the known initial position.
            state["_index_sampler_state"] = {
                "consumed_samples": self._initial_consumed_samples + (
                    self._step * self._planner.local_batch_size * self._topology.data_parallel_size
                ),
                "epoch": self._epoch,
                "global_batch_size": (
                    self._accumulation_steps * self._planner.local_batch_size * self._topology.data_parallel_size
                ),
                "dp_world_size": self._topology.data_parallel_size,
            }
            sampler = state["_index_sampler_state"]
            offset = self._initial_consumed_samples % sampler["global_batch_size"]
            if offset:
                sampler["optimizer_step_offset"] = offset
        try:
            copied_state = copy.deepcopy(state)
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not checkpointable: {exc}") from exc
        return copied_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a shared native cursor or a legacy rank-local checkpoint.

        Args:
            state_dict: A complete-step cursor; legacy rank-local state must match this rank.
        """
        self._validate_restore_boundary()
        try:
            state = copy.deepcopy(dict(state_dict))
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not copyable: {exc}") from exc
        if state.get("global_rank") not in (None, self._topology.global_rank):
            raise ValueError(
                f"Distributed DataLoader checkpoint global_rank={state.get('global_rank')!r} "
                f"does not match {self._topology.global_rank!r}."
            )
        epoch, step, stopped = self._validate_checkpoint_values(state)
        sampler = get_sampler_checkpoint_state(state)
        # Legacy step counters use the source accumulation size. The sampler
        # cursor validates complete steps independently of the target GBS/DP.
        layout_changed = sampler is not None and (
            sampler["global_batch_size"] != (
                self._accumulation_steps * self._planner.local_batch_size * self._topology.data_parallel_size
            ) or sampler["dp_world_size"] != self._topology.data_parallel_size
        )
        if (
                (not layout_changed and step % self._accumulation_steps) or state.get("window_plans")
                or state.get("optimizer_step_remaining", self._accumulation_steps) != self._accumulation_steps
                or any(isinstance(state.get(key), Mapping) and "window" in state[key]
                       for key in ("dataset_reader", "metadata_reader", "sidecar_reader"))
        ):
            raise ValueError("Distributed DataLoader checkpoint must align to a complete optimizer step.")
        if sampler is not None:
            epoch, step = self._progress_from_sampler(sampler)
        elif self._dataset_reader is not None or self._metadata_reader is not None or state.get("global_rank") is None:
            raise ValueError("Distributed DataLoader checkpoint has no native sampler cursor.")
        self._restore_component_states(state, sampler, epoch)
        self._epoch = epoch
        self._step = step
        self._stopped = stopped
        self._initial_consumed_samples = None
        if sampler is not None:
            self._initial_consumed_samples = sampler["consumed_samples"] - (
                step * self._planner.local_batch_size * self._topology.data_parallel_size
            )

        self._delivered_state = self._snapshot_state()

    def _progress_from_sampler(self, sampler: Mapping[str, Any]) -> tuple[int, int]:
        """Validate the shared cursor and derive the epoch and micro-batch position."""
        global_micro_batch_size = self._planner.local_batch_size * self._topology.data_parallel_size
        global_batch_size = self._accumulation_steps * global_micro_batch_size
        consumed, epoch = sampler["consumed_samples"], sampler["epoch"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (consumed, epoch)):
            raise ValueError("Sampler consumed_samples and epoch must be integers")
        if sampler_optimizer_step_progress(sampler) or epoch < 0:
            raise ValueError("Sampler cursor must align to a complete optimizer step")
        if sampler["global_batch_size"] != global_batch_size:
            return epoch, 0
        offset = sampler.get("optimizer_step_offset") or 0
        return epoch, (consumed - offset) // global_micro_batch_size

    def _validate_restore_boundary(self) -> None:
        if self._last_ready_event is not None or self._prefetch_thread is not None or self._step != 0:
            raise ValueError("load_state_dict must run before distributed iteration starts.")

    @staticmethod
    def _validate_checkpoint_values(state: Mapping[str, Any]) -> tuple[int, int, bool]:
        """Validate and return the saved epoch, micro-batch index and stopped flag."""
        epoch = state.get("epoch", 0)
        step = state.get("step", 0)
        stopped = state.get("_iterator_finished")
        if stopped is None:
            stopped = state.get("stopped", False)
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Distributed DataLoader checkpoint epoch must be non-negative.")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Distributed DataLoader checkpoint step must be non-negative.")
        if not isinstance(stopped, bool):
            raise ValueError("Distributed DataLoader checkpoint stopped must be boolean.")
        return epoch, step, stopped

    def _restore_component_states(
            self,
            state: Mapping[str, Any],
            sampler_state: Mapping[str, Any] | None,
            epoch: int,
    ) -> None:
        """Rebuild current Readers from shared progress and optional Dataset checks."""
        reader = self._planning_reader()
        reader_state = reader.state_dict() if reader is not None else None
        payload_state = (
            self._direct_sample_loader.state_dict() if self._direct_sample_loader is not None
            else (reader_state or {}).get("sample_loader")
        )
        # Native torchdata Dataset state belongs to the user's Dataset. Only
        # cursor-only and legacy balanced states carry our Dataset compatibility fields.
        if not has_torchdata_state(state):
            candidates = [state.get("dataset_state"), state.get("direct_sample_loader")]
            for key in ("dataset_reader", "metadata_reader", "sidecar_reader"):
                legacy_reader = state.get(key)
                if isinstance(legacy_reader, Mapping):
                    candidates.append(legacy_reader.get("sample_loader"))
            for saved in candidates:
                if saved is None or payload_state is None:
                    continue
                if not isinstance(saved, Mapping):
                    raise ValueError("Dataset checkpoint state must be a mapping.")
                for key in ("dataset_size", "seed"):
                    if saved.get(key) is not None and saved[key] != payload_state[key]:
                        raise ValueError(f"Dataset checkpoint {key} does not match the current loader.")
        if payload_state is not None:
            payload_state["epoch"] = epoch
        if reader is not None:
            reader_state.update(epoch=epoch, sampler=sampler_state)
            if reader_state["sample_loader"] is not None:
                reader_state["sample_loader"] = payload_state
            reader.load_state_dict(reader_state)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.load_state_dict(payload_state)

    def set_epoch(self, epoch: int) -> None:
        """Reset the native source while retaining the Host thread and device streams.

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
        if self._dataset_reader is not None:
            self._dataset_reader.set_epoch(epoch)
        if self._metadata_reader is not None:
            self._metadata_reader.set_epoch(epoch)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.set_epoch(epoch)
        self._epoch = epoch
        self._step = 0
        self._initial_consumed_samples = 0
        self._stopped = False
        self._pending_local_keys.clear()
        self._window_plans.clear()
        self._current = None
        self._future = None
        self._device_batches.clear()
        self._consumer_stream = None
        self._prefetch_slots = deque([None] * (self._device_prefetch_depth + 1))
        self._device_stopped = False
        self._prefetch_target = 0
        self._batch_error = None
        self._last_ready_event = None
        self._has_delivered = False
        self._balance_logs = []
        self._delivered_state = self._snapshot_state()

    def _produce_on_data_plane(self) -> Any:
        """Execute one plan through direct reads or payload exchange, then construct the batch."""
        plan = self._next_plan_control()
        if plan is None:
            self._stopped = True
            return None

        self._pending_local_keys = {
            key for key in plan.selected_keys if key.reader_rank == self._topology.global_rank
        }
        if self._metadata_mode:
            if not self._topology.is_constructor:
                return None
            received_payloads = self._direct_sample_loader.fetch_samples(
                plan.local_sample_keys(self._topology.data_rank)
            )
        else:
            prepared_exchange = self._prepare_outgoing(plan, self._pending_local_keys)
            received_payloads = self._data_plane.exchange_prepared(prepared_exchange)
            # Reader-only ranks must finish routing before leaving this round.
            if not self._topology.is_constructor:
                return None
        return self._construct_received_payloads(plan, received_payloads)

    def _construct_received_payloads(
            self,
            plan: DistributedPackingPlan,
            received_payloads: dict[SampleKey, Any],
    ) -> Any:
        """Construct one local batch; the caller must be a Data Constructor."""
        local_batch_plan = plan.local_batch_for(self._topology.data_rank)
        local_batch = self._data_constructor.construct(local_batch_plan, received_payloads)
        # None is reserved for distributed end-of-stream.
        if local_batch is None:
            raise ValueError("The Data Constructor must return a non-None batch.")
        return local_batch

    def _next_plan_control(self) -> DistributedPackingPlan | None:
        """Pop one micro-batch plan, planning a new window only when the cache is empty."""
        if self._window_plans:
            return self._window_plans.popleft()
        planning_reader = self._planning_reader()
        local_input = planning_reader.planning_input() if planning_reader is not None else _ReaderPlanInput()
        if self._broadcast_plan:
            gathered_inputs = self._data_plane.gather_object_to_planner(local_input)
            plan = None
            if self._topology.global_rank == self._data_plane.planner_rank:
                plan = self._build_plan_control(gathered_inputs)
            plan = self._data_plane.broadcast_from_planner(plan)
        else:
            plan = self._build_plan_control(self._data_plane.all_gather_object(local_input))
        if isinstance(plan, tuple):
            self._window_plans.extend(plan)
            return self._window_plans.popleft()
        return plan

    def _prepare_local_metadata(self, window_micro_batch_num: int) -> None:
        """Read one explicitly sized source window before constructing its batches."""
        if self._topology.global_rank not in self._dataset_reader_ranks:
            return
        planning_reader = self._planning_reader()
        planning_reader.prepare_next_step(window_micro_batch_num)

    def _build_plan_control(
            self, gathered_inputs: tuple[_ReaderPlanInput, ...],
    ) -> _PlanControl:
        """Return the native window's plans; None means end of stream."""
        # Collective rank order can differ from the mesh's logical DP order.
        inputs_by_rank = dict(zip(self._data_plane.ranks, gathered_inputs))
        reader_inputs = [
            inputs_by_rank[rank] for rank in self._topology.constructor_ranks
            if rank in self._dataset_reader_ranks
        ]
        return self._plan_native_batch(reader_inputs)

    def _native_window_micro_batch_num(self, reader_inputs: list[_ReaderPlanInput]) -> int:
        """Validate complete synchronized source rounds before constructing reference bins."""
        if all(not item.original_batches for item in reader_inputs):
            return 0
        if any(not item.original_batches for item in reader_inputs):
            raise ValueError("Native BatchSampler ranks exhausted at different forward/backward rounds.")
        counts = {len(item.original_batches) for item in reader_inputs}
        if len(counts) != 1 or any(
                len(batch) != self._planner.local_batch_size
                for item in reader_inputs for batch in item.original_batches
        ):
            raise ValueError("Native BatchSampler ranks must provide equal complete micro-batch windows.")
        return next(iter(counts))

    def _plan_native_batch(
            self, reader_inputs: list[_ReaderPlanInput],
    ) -> _PlanControl:
        """Plan the native sampler window, returning one plan per training round."""
        window_micro_batch_num = self._native_window_micro_batch_num(reader_inputs)
        if not window_micro_batch_num:
            return None
        reference_bins = tuple(
            keys for batches in zip(*(item.original_batches for item in reader_inputs))
            for batch in batches for keys in batch
        )
        metadata_by_key = {metadata.key: metadata for item in reader_inputs for metadata in item.metadata}
        samples = tuple(metadata_by_key[key] for keys in reference_bins for key in keys)
        plan = self._planner.plan if window_micro_batch_num == 1 else self._planner.plan_window
        plans = plan(samples, reference_bins=reference_bins, step=self._step)
        self._capture_balance_log(samples, reference_bins, plans)
        return plans

    def _capture_balance_log(
            self, samples: tuple[SampleMetadata, ...], reference_bins: tuple[tuple[SampleKey, ...], ...],
            plans: DistributedPackingPlan | tuple[DistributedPackingPlan, ...],
    ) -> None:
        """Freeze scalar window statistics before the producer advances to another plan."""
        if (
            not self._log_balance
            or self._topology.global_rank != self._data_plane.planner_rank
            or not balance_logger.isEnabledFor(logging.INFO)
        ):
            return
        plans = plans if isinstance(plans, tuple) else (plans,)
        metadata = {item.key: item for item in samples}
        costs = self._planner.last_sample_costs

        def describe(keys: tuple[SampleKey, ...]) -> dict[str, Any]:
            """Combine evaluated document costs with this layout's constructor padding.

            Args:
                keys: Sample occurrences assigned to one pack.
            """
            tokens = sum(metadata[key].pack_tokens for key in keys)
            padding = max(0, self._planner.seq_len - tokens)
            return {
                "seq_len": tokens, "samples": len(keys), "padding": padding,
                "cost": math.fsum(costs[key].llm for key in keys) + self._planner.padding_cost(padding).llm,
            }

        local_size = self._planner.local_batch_size
        bin_count = self._planner.distributed_bin_count
        rounds = []
        for offset, plan in enumerate(plans):
            original = reference_bins[offset * bin_count:(offset + 1) * bin_count]
            rounds.append({
                "micro_batch": plan.step % self._accumulation_steps + 1,
                "bins_before": tuple(
                    tuple(describe(keys) for keys in original[start:start + local_size])
                    for start in range(0, bin_count, local_size)
                ),
                "bins_after": tuple(
                    tuple(describe(packing_bin.sample_keys) for packing_bin in local_batch)
                    for local_batch in plan.local_batches
                ),
            })
        stats = {
            **self._planner.last_balance_decision,
            "objective": self._planner.objective, "epoch": self._epoch,
            "planner_rank": self._data_plane.planner_rank,
            "constructor_ranks": self._topology.constructor_ranks,
            "group_ranks": tuple(range(self._topology.data_parallel_size)), "rounds": rounds,
        }
        self._balance_logs.append(format_balance_stats(stats, self._step // self._accumulation_steps + 1))

    def _prepare_outgoing(
            self,
            plan: DistributedPackingPlan,
            local_selected_keys: set[SampleKey],
    ) -> PreparedPayloadExchange:
        """Group source-owned payloads by destination Constructor for transport encoding."""
        payloads_by_target_rank: dict[int, list[tuple[SampleKey, Any]]] = {}
        if local_selected_keys:
            payloads = self._dataset_reader.selected_payloads(local_selected_keys)
            target_by_key = {
                key: self._topology.constructor_ranks[data_rank]
                for data_rank, local_batch in enumerate(plan.local_batches)
                for packing_bin in local_batch
                for key in packing_bin.sample_keys
            }
            for key, payload in payloads:
                payloads_by_target_rank.setdefault(target_by_key[key], []).append((key, payload))
        return self._data_plane.prepare_exchange(payloads_by_target_rank)

    def _planning_reader(self) -> Any | None:
        """Return this rank's online or metadata-only Dataset Reader."""
        if self._dataset_reader is not None:
            return self._dataset_reader
        return self._metadata_reader



__all__ = ["DistributedDataLoader"]
