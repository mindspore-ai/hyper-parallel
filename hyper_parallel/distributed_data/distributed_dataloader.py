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
import pickle
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from threading import Thread
from typing import Any, Literal

from hyper_parallel.distributed_data.data_constructor import PackingDataConstructor
from hyper_parallel.distributed_data.batch_sampler import BatchSamplerReader
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    ConstructedBatch,
    DistributedPackingPlan,
    SampleKey,
    StepSampleSelection,
)
from hyper_parallel.distributed_data.step_sample_selection import StepSampleSelector
from hyper_parallel.distributed_data.metadata import MetadataReader, PlannedSampleLoader
from hyper_parallel.distributed_data.dataset_reader import DatasetReader
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    PreparedPayloadExchange,
)

_ControlKind = Literal["plan", "need_more", "stop", "error"]


def _scaled_buffer_target(
        base_target: int,
        max_target: int,
        multiplier: int | float,
        attempt: int,
        reader_count: int,
) -> int:
    """Scale and saturate a read-ahead target without floating-point overflow."""
    multiplier_numerator, multiplier_denominator = multiplier.as_integer_ratio()
    denominator = multiplier_denominator * reader_count
    numerator = base_target * multiplier_numerator * attempt
    if numerator >= max_target * denominator:
        return max_target
    return max(1, (numerator + denominator - 1) // denominator)


@dataclass(frozen=True)
class _ReaderSnapshot:
    rank: int
    step: int
    stopped: bool
    is_reader: bool
    exhausted: bool
    can_read_more: bool
    metadata: tuple[BufferedSampleMetadata, ...]
    error: str | None = None
    batch_position: int | None = None


@dataclass(frozen=True)
class _PlanControl:
    kind: _ControlKind
    plan: DistributedPackingPlan | None = None
    error: str | None = None


@dataclass(frozen=True)
class _PrefetchResult:
    delivery: ConstructedBatch | None = None
    error: BaseException | None = None


def _validate_loader_components(
        *,
        topology: DataTopology,
        dataset_reader_ranks: tuple[int, ...],
        dataset_reader: DatasetReader | BatchSamplerReader | None,
        metadata_reader: MetadataReader | BatchSamplerReader | None,
        direct_sample_loader: PlannedSampleLoader | None,
        metadata_mode: bool,
        metadata_payload_exchange: bool,
        double_buffer: bool,
) -> None:
    if not isinstance(metadata_mode, bool):
        raise ValueError("metadata_mode must be boolean.")
    if not isinstance(metadata_payload_exchange, bool):
        raise ValueError("metadata_payload_exchange must be boolean.")
    if metadata_payload_exchange and not metadata_mode:
        raise ValueError("metadata_payload_exchange requires metadata mode.")
    if metadata_mode and dataset_reader is not None:
        raise ValueError("Metadata mode must not configure an online Dataset Reader.")
    if not metadata_mode and (metadata_reader is not None or direct_sample_loader is not None):
        raise ValueError("Online mode must not configure metadata loading components.")
    is_reader = topology.global_rank in dataset_reader_ranks
    planning_reader = metadata_reader if metadata_mode else dataset_reader
    if is_reader != (planning_reader is not None):
        raise ValueError("Dataset Reader ownership does not match dataset_reader_ranks.")
    _validate_metadata_loader_owner(
        topology,
        is_reader=is_reader,
        metadata_mode=metadata_mode,
        metadata_payload_exchange=metadata_payload_exchange,
        direct_sample_loader=direct_sample_loader,
    )
    if not isinstance(double_buffer, bool):
        raise ValueError("double_buffer must be boolean.")


def _validate_metadata_loader_owner(
        topology: DataTopology,
        *,
        is_reader: bool,
        metadata_mode: bool,
        metadata_payload_exchange: bool,
        direct_sample_loader: PlannedSampleLoader | None,
) -> None:
    if not metadata_mode:
        return
    expected_loader_owner = is_reader if metadata_payload_exchange else topology.is_constructor
    if expected_loader_owner != (direct_sample_loader is not None):
        owner_name = "Dataset Reader" if metadata_payload_exchange else "Data Constructor"
        raise ValueError(f"Every metadata {owner_name} must own one plan-aware sample loader.")


class DistributedDataLoader(Iterator[Any]):
    """Yield dynamically packed local batches on every training rank.

    One distributed transaction fills Dataset Reader buffers, freezes the
    current Step Sample Selection, balances that exact set, moves payloads,
    constructs local batches, and broadcasts them to model-parallel peers.
    Optional double buffering runs the next transaction in a background thread
    after the current batch has been delivered to the trainer.
    """

    VERSION = 5
    collective_source = True

    def __init__(
            self,
            *,
            topology: DataTopology,
            dataset_reader_ranks: tuple[int, ...],
            dataset_reader: DatasetReader | BatchSamplerReader | None,
            metadata_reader: MetadataReader | BatchSamplerReader | None,
            direct_sample_loader: PlannedSampleLoader | None,
            metadata_mode: bool,
            metadata_payload_exchange: bool,
            step_sample_selector: StepSampleSelector | None,
            planner: DynamicPackingPlanner,
            data_constructor: PackingDataConstructor,
            data_plane: DataPlaneTransport,
            model_transport: ModelParallelTransport,
            buffer_size_multiplier: float,
            max_buffered_samples: int,
            double_buffer: bool,
            config_fingerprint: str,
            batch_sampler_mode: bool = False,
            initial_epoch: int = 0,
    ) -> None:
        """Store the fully validated runtime components."""
        if not batch_sampler_mode and step_sample_selector is None:
            raise ValueError("Stream-based loading requires a StepSampleSelector.")
        _validate_loader_components(
            topology=topology,
            dataset_reader_ranks=dataset_reader_ranks,
            dataset_reader=dataset_reader,
            metadata_reader=metadata_reader,
            direct_sample_loader=direct_sample_loader,
            metadata_mode=metadata_mode,
            metadata_payload_exchange=metadata_payload_exchange,
            double_buffer=double_buffer,
        )
        self._topology = topology
        self._dataset_reader_ranks = dataset_reader_ranks
        self._dataset_reader = dataset_reader
        self._metadata_reader = metadata_reader
        self._direct_sample_loader = direct_sample_loader
        self._metadata_mode = metadata_mode
        self._metadata_payload_exchange = metadata_payload_exchange
        self._step_sample_selector = step_sample_selector
        self._planner = planner
        self._data_constructor = data_constructor
        self._data_plane = data_plane
        self._model_transport = model_transport
        self._buffer_size_multiplier = buffer_size_multiplier
        self._max_buffered_samples = max_buffered_samples
        self._double_buffer = double_buffer
        self._config_fingerprint = config_fingerprint
        self._batch_sampler_mode = batch_sampler_mode
        self._epoch = initial_epoch
        self._step = 0
        self._stopped = False
        self._last_plan_id: str | None = None
        self._last_plan: DistributedPackingPlan | None = None
        self._pending_local_keys: set[SampleKey] = set()
        self._pending_plan: DistributedPackingPlan | None = None
        self._prefetch_thread: Thread | None = None
        self._prefetch_result: _PrefetchResult | None = None

    def __iter__(self) -> "DistributedDataLoader":
        """Return this stateful distributed iterator."""
        return self

    def __next__(self) -> Any:
        """Collectively construct and return the next rank-local batch."""
        received = self._take_prefetched() if self._double_buffer else self._collect_next_delivery()
        data = self._consume_delivery(received)
        if self._double_buffer:
            self._start_prefetch()
        return data

    def _collect_next_delivery(self) -> ConstructedBatch:
        """Run one complete distributed data transaction."""
        was_stopped = self._stopped
        model_group_error = self._model_transport.synchronize_iterator_state(
            epoch=self._epoch,
            step=self._step,
            stopped=self._stopped,
        )
        constructor_delivery = None
        if self._data_plane.is_member:
            iterator_state_error = self._data_plane.synchronize_iterator_state(
                epoch=self._epoch,
                step=self._step,
                stopped=self._stopped,
                model_group_error=model_group_error,
            )
            if iterator_state_error is not None:
                constructor_delivery = self._constructor_envelope(error=iterator_state_error)
            else:
                constructor_delivery = self._produce_on_data_plane()
        received = self._model_transport.broadcast(constructor_delivery)
        if was_stopped and not received.stopped:
            raise ValueError("Distributed DataLoader stopped state differs across model-parallel peers.")
        return received

    def _consume_delivery(self, received: ConstructedBatch) -> Any:
        """Validate, commit, and expose a completed distributed transaction."""
        if received.error is not None:
            raise RuntimeError(received.error)
        if received.step != self._step:
            raise ValueError(
                f"Constructed batch step mismatch: expected {self._step}, got {received.step}."
            )
        if received.stopped:
            self._stopped = True
            self._pending_plan = None
            raise StopIteration
        if not received.plan_id:
            raise ValueError("An active constructed batch must include a plan_id.")
        if self._data_plane.is_member:
            if self._pending_plan is None or self._pending_plan.plan_id != received.plan_id:
                raise ValueError("Delivered batch does not match the pending distributed packing plan.")

        # Dataset Reader buffers are committed only after construction and delivery
        # have both succeeded, leaving checkpoint boundaries unambiguous.
        planning_reader = self._planning_reader()
        if planning_reader is not None:
            planning_reader.commit(self._pending_local_keys)
        self._pending_local_keys.clear()
        self._last_plan_id = received.plan_id
        self._last_plan = self._pending_plan
        self._pending_plan = None
        self._stopped = False
        self._step += 1
        return received.data

    def _start_prefetch(self) -> None:
        """Start one background transaction for the current iterator step."""
        if self._prefetch_thread is not None:
            raise ValueError("A distributed local-batch prefetch is already in flight.")
        self._prefetch_result = None
        self._prefetch_thread = Thread(
            target=self._run_prefetch,
            name=f"hp-data-prefetch-rank-{self._topology.global_rank}",
            daemon=True,
        )
        self._prefetch_thread.start()

    def _run_prefetch(self) -> None:
        """Produce one result without allowing exceptions to strand the consumer."""
        try:
            self._prefetch_result = _PrefetchResult(delivery=self._collect_next_delivery())
        except BaseException as exc:  # The foreground re-raises failures at the next iterator boundary.
            self._prefetch_result = _PrefetchResult(error=exc)

    def _take_prefetched(self) -> ConstructedBatch:
        """Wait for and return the current background result."""
        if self._prefetch_thread is None:
            self._start_prefetch()
        prefetch_thread = self._prefetch_thread
        if prefetch_thread is None:
            raise ValueError("Double buffering did not create a prefetch thread.")
        prefetch_thread.join()
        self._prefetch_thread = None
        result = self._prefetch_result
        self._prefetch_result = None
        if result is None:
            raise ValueError("Double buffering completed without a prefetch result.")
        if result.error is not None:
            raise result.error
        if result.delivery is None:
            raise ValueError("Double buffering completed without a constructed batch delivery.")
        return result.delivery

    @property
    def last_plan_id(self) -> str | None:
        """Return the most recently delivered deterministic plan identifier."""
        return self._last_plan_id

    def wait_for_prefetch(self) -> None:
        """Wait for in-flight data communication before training tears down groups.

        All ranks must call this at the same consumed-batch boundary. The
        prepared result remains available to the next iterator call.
        """
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()

    @property
    def last_plan(self) -> DistributedPackingPlan | None:
        """Return the last plan on data-plane ranks, otherwise ``None``."""
        return self._last_plan

    def state_dict(self) -> dict[str, Any]:
        """Return rank-local state at a completed distributed-batch boundary."""
        restart_prefetch = self._prefetch_thread is not None
        if restart_prefetch:
            self._take_prefetched()
            # Prefetch is speculative until the trainer requests the batch.
            # Keep Reader buffers uncommitted so the checkpoint can replan it.
            self._pending_local_keys.clear()
            self._pending_plan = None
        if self._pending_local_keys:
            raise ValueError("Cannot checkpoint while a distributed batch is in flight.")
        state = {
            "version": self.VERSION,
            "topology_fingerprint": self._topology.fingerprint,
            "config_fingerprint": self._config_fingerprint,
            "global_rank": self._topology.global_rank,
            "epoch": self._epoch,
            "step": self._step,
            "stopped": self._stopped,
            "last_plan_id": self._last_plan_id,
            "dataset_reader": self._dataset_reader.state_dict() if self._dataset_reader is not None else None,
            "metadata_reader": self._metadata_reader.state_dict() if self._metadata_reader is not None else None,
            "direct_sample_loader": (
                self._direct_sample_loader.state_dict() if self._direct_sample_loader is not None else None
            ),
        }
        try:
            copied_state = copy.deepcopy(state)
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not checkpointable: {exc}") from exc
        if restart_prefetch:
            self._start_prefetch()
        return copied_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a fixed-topology rank-local checkpoint.

        Args:
            state_dict: State produced on this same global rank.
        """
        self._validate_restore_boundary()
        try:
            state = copy.deepcopy(dict(state_dict))
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not copyable: {exc}") from exc
        self._validate_checkpoint_identity(state)
        epoch, step, stopped, last_plan_id = self._validate_checkpoint_values(state)
        reader_state, metadata_reader_state, direct_sample_state = self._validate_component_states(state, epoch)
        self._restore_component_states(reader_state, metadata_reader_state, direct_sample_state)
        self._epoch = epoch
        self._step = step
        self._stopped = stopped
        self._last_plan_id = last_plan_id

    def _validate_restore_boundary(self) -> None:
        active_state = (
            self._step != 0
            or bool(self._pending_local_keys)
            or self._pending_plan is not None
            or self._prefetch_thread is not None
        )
        if active_state:
            raise ValueError("load_state_dict must run before distributed iteration starts.")

    def _validate_checkpoint_identity(self, state: Mapping[str, Any]) -> None:
        expected = {
            "version": self.VERSION,
            "topology_fingerprint": self._topology.fingerprint,
            "config_fingerprint": self._config_fingerprint,
            "global_rank": self._topology.global_rank,
        }
        for name, expected_value in expected.items():
            if state.get(name) != expected_value:
                raise ValueError(
                    f"Distributed DataLoader checkpoint {name}={state.get(name)!r} "
                    f"does not match {expected_value!r}."
                )

    @staticmethod
    def _validate_checkpoint_values(state: Mapping[str, Any]) -> tuple[int, int, bool, str | None]:
        epoch = state.get("epoch")
        step = state.get("step")
        stopped = state.get("stopped")
        last_plan_id = state.get("last_plan_id")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Distributed DataLoader checkpoint epoch must be non-negative.")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Distributed DataLoader checkpoint step must be non-negative.")
        if not isinstance(stopped, bool):
            raise ValueError("Distributed DataLoader checkpoint stopped must be boolean.")
        if last_plan_id is not None and (not isinstance(last_plan_id, str) or not last_plan_id):
            raise ValueError("Distributed DataLoader checkpoint last_plan_id is invalid.")
        return epoch, step, stopped, last_plan_id

    def _validate_component_states(
            self,
            state: Mapping[str, Any],
            epoch: int,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any] | None]:
        reader_state = state.get("dataset_reader")
        # Accept the old field name when restoring checkpoints written before the rename.
        metadata_reader_state = state.get("metadata_reader", state.get("sidecar_reader"))
        direct_sample_state = state.get("direct_sample_loader")
        if (self._dataset_reader is None) != (reader_state is None):
            raise ValueError("Distributed DataLoader checkpoint Dataset Reader ownership changed.")
        if (self._metadata_reader is None) != (metadata_reader_state is None):
            raise ValueError("Distributed DataLoader checkpoint metadata ownership changed.")
        if (self._direct_sample_loader is None) != (direct_sample_state is None):
            raise ValueError("Distributed DataLoader checkpoint direct-reader ownership changed.")
        for component_name, component_state in (
                ("Dataset Reader", reader_state),
                ("metadata reader", metadata_reader_state),
                ("direct reader", direct_sample_state),
        ):
            if component_state is None:
                continue
            if not isinstance(component_state, Mapping):
                raise ValueError(f"Distributed DataLoader checkpoint {component_name} state must be a mapping.")
            if component_state.get("epoch") != epoch:
                raise ValueError(
                    f"Distributed DataLoader checkpoint {component_name} epoch "
                    f"{component_state.get('epoch')!r} does not match loader epoch {epoch}."
                )
        return reader_state, metadata_reader_state, direct_sample_state

    def _restore_component_states(
            self,
            reader_state: Mapping[str, Any] | None,
            metadata_reader_state: Mapping[str, Any] | None,
            direct_sample_state: Mapping[str, Any] | None,
    ) -> None:
        if self._dataset_reader is not None:
            self._dataset_reader.load_state_dict(reader_state)
        if self._metadata_reader is not None:
            self._metadata_reader.load_state_dict(metadata_reader_state)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.load_state_dict(direct_sample_state)

    def set_epoch(self, epoch: int) -> None:
        """Reset an exhausted loader for a deterministic new Dataset epoch.

        Args:
            epoch: Non-negative Dataset epoch.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, but got {epoch!r}.")
        if self._step != 0 and not self._stopped:
            raise ValueError("set_epoch requires a fresh or exhausted Distributed DataLoader.")
        if self._prefetch_thread is not None:
            raise ValueError("set_epoch cannot run while a double-buffer prefetch is in flight.")
        if self._dataset_reader is not None:
            self._dataset_reader.set_epoch(epoch)
        if self._metadata_reader is not None:
            self._metadata_reader.set_epoch(epoch)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.set_epoch(epoch)
        self._epoch = epoch
        self._step = 0
        self._stopped = False
        self._last_plan_id = None
        self._last_plan = None
        self._pending_plan = None

    def _produce_on_data_plane(self) -> ConstructedBatch | None:
        control = self._next_plan_control()
        if control.kind == "stop":
            return self._constructor_envelope(stopped=True)
        if control.kind == "error":
            return self._constructor_envelope(error=control.error or "Distributed data planning failed.")
        if control.kind != "plan" or control.plan is None:
            return self._constructor_envelope(error="Planner returned an invalid control message.")

        plan = control.plan
        if plan.step != self._step:
            return self._constructor_envelope(
                error=f"Planner returned step {plan.step}, but this rank expects step {self._step}."
            )
        self._pending_plan = plan
        selected_keys = set(plan.selected_keys)
        local_selected_keys = {
            key for key in selected_keys if key.reader_rank == self._topology.global_rank
        }
        self._pending_local_keys = local_selected_keys
        if self._metadata_mode and not self._metadata_payload_exchange:
            return self._produce_metadata_batch(plan)

        outgoing, preparation_error = self._prepare_outgoing(plan, local_selected_keys)
        shared_error = self._data_plane.synchronize_error(preparation_error)
        if shared_error is not None:
            self._pending_local_keys.clear()
            return self._constructor_envelope(error=shared_error)
        if outgoing is None:
            self._pending_local_keys.clear()
            return self._constructor_envelope(error="Payload preflight returned no prepared exchange.")

        received_payloads: dict[SampleKey, Any] = {}
        exchange_error = None
        try:
            received_payloads = self._data_plane.exchange_prepared(outgoing)
        except Exception as exc:  # A2A has completed; synchronize decode/validation errors next.
            exchange_error = self._format_error("sample payload exchange", exc)
        shared_error = self._data_plane.synchronize_error(exchange_error)
        if shared_error is not None:
            self._pending_local_keys.clear()
            return self._constructor_envelope(error=shared_error)

        return self._construct_received_payloads(plan, received_payloads)

    def _produce_metadata_batch(self, plan: DistributedPackingPlan) -> ConstructedBatch | None:
        """Directly read constructor-assigned shared indices without payload A2A."""
        received_payloads: dict[SampleKey, Any] = {}
        fetch_error = None
        if self._topology.is_constructor:
            try:
                if self._direct_sample_loader is None:
                    raise ValueError("A metadata Data Constructor has no plan-aware sample loader.")
                constructor_plan = plan.constructor_for(self._topology.data_rank)
                received_payloads = self._direct_sample_loader.fetch(constructor_plan)
            except Exception as exc:
                fetch_error = self._format_error("metadata direct read", exc)
        shared_error = self._data_plane.synchronize_error(fetch_error)
        if shared_error is not None:
            self._pending_local_keys.clear()
            return self._constructor_envelope(error=shared_error)
        return self._construct_received_payloads(plan, received_payloads)

    def _construct_received_payloads(
            self,
            plan: DistributedPackingPlan,
            received_payloads: dict[SampleKey, Any],
    ) -> ConstructedBatch | None:
        """Construct one local batch and synchronize constructor failures."""
        local_batch = None
        construction_error = None
        if self._topology.is_constructor:
            try:
                constructor_plan = plan.constructor_for(self._topology.data_rank)
                local_batch = self._data_constructor.construct(constructor_plan, received_payloads)
                # Object broadcast uses pickle; fail collectively before peers
                # enter different model-group collectives.
                pickle.dumps(local_batch, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                construction_error = self._format_error("Data Constructor", exc)
        elif received_payloads:
            construction_error = (
                f"Non-constructor rank {self._topology.global_rank} received unexpected sample payloads."
            )
        shared_error = self._data_plane.synchronize_error(construction_error)
        if shared_error is not None:
            self._pending_local_keys.clear()
            return self._constructor_envelope(error=shared_error)
        if not self._topology.is_constructor:
            return None
        return ConstructedBatch(step=plan.step, plan_id=plan.plan_id, data=local_batch)

    def _next_plan_control(self) -> _PlanControl:
        attempt = 1
        while True:
            local_snapshot = self._fill_local_reader(attempt)
            snapshots = self._data_plane.gather_object_to_planner(local_snapshot)
            planner_control = None
            if self._topology.global_rank == self._data_plane.planner_rank:
                if snapshots is None:
                    planner_control = _PlanControl("error", error="Planner did not receive Dataset Reader snapshots.")
                else:
                    planner_control = self._build_plan_control(snapshots)
            control = self._data_plane.broadcast_from_planner(planner_control)
            if not isinstance(control, _PlanControl):
                return _PlanControl("error", error="Planner broadcast an invalid control message.")
            if control.kind != "need_more":
                return control
            attempt += 1

    def _fill_local_reader(self, attempt: int) -> _ReaderSnapshot:
        is_reader = self._topology.global_rank in self._dataset_reader_ranks
        if not is_reader:
            return _ReaderSnapshot(
                rank=self._topology.global_rank,
                step=self._step,
                stopped=self._stopped,
                is_reader=False,
                exhausted=True,
                can_read_more=False,
                metadata=(),
            )
        planning_reader = self._planning_reader()
        if planning_reader is None:
            return _ReaderSnapshot(
                rank=self._topology.global_rank,
                step=self._step,
                stopped=self._stopped,
                is_reader=True,
                exhausted=False,
                can_read_more=False,
                metadata=(),
                error=f"Dataset Reader rank {self._topology.global_rank} did not provide its reader.",
            )

        reader_count = len(self._dataset_reader_ranks)
        sample_target = _scaled_buffer_target(
            self._planner.distributed_bin_count,
            self._max_buffered_samples,
            self._buffer_size_multiplier,
            attempt,
            reader_count,
        )
        token_target = _scaled_buffer_target(
            self._planner.distributed_token_budget,
            self._max_buffered_samples * self._planner.seq_len,
            self._buffer_size_multiplier,
            attempt,
            reader_count,
        )
        error = planning_reader.fill(
            min_samples=max(1, sample_target),
            min_tokens=max(1, token_target),
            max_samples=self._max_buffered_samples,
        )
        return _ReaderSnapshot(
            rank=self._topology.global_rank,
            step=self._step,
            stopped=self._stopped,
            is_reader=True,
            exhausted=planning_reader.exhausted,
            can_read_more=(
                not planning_reader.exhausted
                and planning_reader.buffer_size < self._max_buffered_samples
            ),
            metadata=planning_reader.metadata(),
            error=error,
            batch_position=planning_reader.batch_position if isinstance(planning_reader, BatchSamplerReader) else None,
        )

    def _build_plan_control(self, snapshots: tuple[Any, ...]) -> _PlanControl:
        normalized, validation_error = self._normalize_reader_snapshots(snapshots)
        if validation_error is not None:
            return validation_error
        state_control = self._snapshot_state_control(normalized)
        if state_control is not None:
            return state_control
        return self._plan_reader_snapshots(normalized)

    def _normalize_reader_snapshots(
            self,
            snapshots: tuple[Any, ...],
    ) -> tuple[tuple[_ReaderSnapshot, ...], _PlanControl | None]:
        normalized = []
        for snapshot in snapshots:
            if not isinstance(snapshot, _ReaderSnapshot):
                error = _PlanControl("error", error="A data-plane rank contributed an invalid reader snapshot.")
                return (), error
            expected_reader = snapshot.rank in self._dataset_reader_ranks
            if snapshot.is_reader != expected_reader:
                error = _PlanControl(
                    "error",
                    error=f"Rank {snapshot.rank} reported inconsistent Dataset Reader ownership.",
                )
                return (), error
            normalized.append(snapshot)
        contributed_ranks = [snapshot.rank for snapshot in normalized]
        if len(contributed_ranks) != len(set(contributed_ranks)) or set(contributed_ranks) != set(
                self._data_plane.ranks
        ):
            error = _PlanControl(
                "error",
                error=f"Data-plane reader snapshots have invalid rank coverage {contributed_ranks}.",
            )
            return (), error
        errors = sorted((snapshot.rank, snapshot.error) for snapshot in normalized if snapshot.error is not None)
        if errors:
            return (), _PlanControl("error", error=errors[0][1])
        return tuple(normalized), None

    def _snapshot_state_control(self, normalized: tuple[_ReaderSnapshot, ...]) -> _PlanControl | None:
        steps = {snapshot.step for snapshot in normalized}
        stopped_states = {snapshot.stopped for snapshot in normalized}
        if len(steps) != 1 or self._step not in steps:
            return _PlanControl(
                "error",
                error=f"Data-plane ranks have inconsistent checkpoint steps {sorted(steps)}.",
            )
        if len(stopped_states) != 1:
            return _PlanControl("error", error="Data-plane ranks have inconsistent stopped checkpoint state.")
        if stopped_states == {True}:
            return _PlanControl("stop")
        return None

    def _plan_reader_snapshots(self, normalized: tuple[_ReaderSnapshot, ...]) -> _PlanControl:
        reader_snapshots = [snapshot for snapshot in normalized if snapshot.is_reader]
        candidates = tuple(metadata for snapshot in reader_snapshots for metadata in snapshot.metadata)
        try:
            if self._batch_sampler_mode:
                selection = self._select_native_batch(reader_snapshots)
            else:
                selection = self._step_sample_selector.select(
                    candidates,
                    end_of_stream=all(snapshot.exhausted for snapshot in reader_snapshots),
                )
            plan = None if selection is None else self._planner.plan(selection, step=self._step)
        except Exception as exc:
            return _PlanControl("error", error=self._format_error("Step Sample Selection or Planner", exc))
        if plan is not None:
            return _PlanControl("plan", plan=plan)
        if all(snapshot.exhausted for snapshot in reader_snapshots):
            return _PlanControl("stop")
        if any(snapshot.can_read_more for snapshot in reader_snapshots):
            return _PlanControl("need_more")
        return _PlanControl(
            "error",
            error=(
                f"Dataset Reader buffers reached max_buffered_samples={self._max_buffered_samples} before "
                f"Step Sample Selection could form {self._planner.distributed_bin_count} complete packing bins."
            ),
        )

    def _select_native_batch(self, snapshots: list[_ReaderSnapshot]) -> StepSampleSelection | None:
        """Freeze this forward/backward round exactly as native DP samplers selected it."""
        positions = {snapshot.batch_position for snapshot in snapshots}
        if len(positions) != 1 or None in positions:
            raise ValueError("Native BatchSampler ranks have inconsistent consumed_samples cursors.")
        if all(snapshot.exhausted for snapshot in snapshots):
            return None
        if any(snapshot.exhausted for snapshot in snapshots):
            raise ValueError("Native BatchSampler ranks exhausted at different forward/backward rounds.")
        if any(len(snapshot.metadata) != self._planner.local_batch_size for snapshot in snapshots):
            raise ValueError("Native BatchSampler ranks must each provide exactly local_batch_size samples.")
        samples = tuple(sorted(
            (item for snapshot in snapshots for item in snapshot.metadata),
            key=lambda item: item.global_sample_position,
        ))
        return StepSampleSelection(samples=samples, reference_bins=tuple((item.key,) for item in samples))

    def _prepare_outgoing(
            self,
            plan: DistributedPackingPlan,
            local_selected_keys: set[SampleKey],
    ) -> tuple[PreparedPayloadExchange | None, str | None]:
        outgoing: dict[int, list[tuple[SampleKey, Any]]] = {}
        try:
            if local_selected_keys:
                if self._metadata_payload_exchange:
                    if self._direct_sample_loader is None:
                        raise ValueError("A pre-sharded metadata Reader has no plan-aware sample loader.")
                    ordered_keys = tuple(key for key in plan.selected_keys if key in local_selected_keys)
                    payloads = tuple(self._direct_sample_loader.fetch_keys(ordered_keys).items())
                else:
                    if self._dataset_reader is None:
                        raise ValueError("A planned Dataset Reader rank has no Dataset Reader.")
                    payloads = self._dataset_reader.selected_payloads(local_selected_keys)
                target_by_key = {
                    sample.key: self._topology.constructor_ranks[constructor.target_data_rank]
                    for constructor in plan.constructors
                    for packing_bin in constructor.bins
                    for sample in packing_bin.samples
                }
                for key, payload in payloads:
                    outgoing.setdefault(target_by_key[key], []).append((key, payload))
            prepared = self._data_plane.prepare_exchange(outgoing)
            return prepared, None
        except Exception as exc:
            return None, self._format_error("payload serialization/allocation", exc)

    def _planning_reader(self) -> DatasetReader | MetadataReader | BatchSamplerReader | None:
        """Return this rank's online or metadata-only Dataset Reader."""
        if self._dataset_reader is not None:
            return self._dataset_reader
        return self._metadata_reader

    def _constructor_envelope(
            self,
            *,
            stopped: bool = False,
            error: str | None = None,
    ) -> ConstructedBatch | None:
        self._pending_local_keys.clear()
        self._pending_plan = None
        if not self._topology.is_constructor:
            return None
        return ConstructedBatch(step=self._step, plan_id=None, stopped=stopped, error=error)

    @staticmethod
    def _format_error(stage: str, exception: Exception) -> str:
        return f"{stage} failed: {type(exception).__name__}: {exception}"


__all__ = ["DistributedDataLoader"]
