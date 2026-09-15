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
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.data_constructor import PackingDataConstructor
from hyper_parallel.distributed_data.batch_sampler import BatchSamplerReader
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    SampleKey,
    StepSampleSelection,
)
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    PreparedPayloadExchange,
)


@dataclass(frozen=True)
class _ReaderSnapshot:
    rank: int
    exhausted: bool
    metadata: tuple[BufferedSampleMetadata, ...]
    # External-step readers provide the legacy producer's already selected
    # local pack boundaries. Native BatchSampler uses singleton bins instead.
    original_metadatas: tuple[tuple[BufferedSampleMetadata, ...], ...] = ()


_PREFETCH_PENDING = object()


def _validate_loader_components(
        *,
        topology: DataTopology,
        dataset_reader_ranks: tuple[int, ...],
        dataset_reader: Any | None,
        metadata_reader: BatchSamplerReader | None,
        direct_sample_loader: PlannedSampleLoader | None,
        metadata_mode: bool,
        double_buffer: bool,
) -> None:
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
    if not isinstance(double_buffer, bool):
        raise ValueError("double_buffer must be boolean.")


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


class DistributedDataLoader(Iterator[Any]):
    """Yield dynamically packed local batches on every training rank.

    One distributed transaction fills Dataset Reader buffers, freezes the
    current Step Sample Selection, balances that exact set, moves payloads,
    constructs local batches, and broadcasts them to model-parallel peers.
    Optional double buffering runs the next transaction in a background thread
    after the current batch has been returned to the trainer.
    """

    # HP Trainer and batch adapters must advance this loader on every model peer.
    collective_source = True

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
            double_buffer: bool,
            config_fingerprint: str,
            batch_sampler_mode: bool = False,
            external_step_mode: bool = False,
            initial_epoch: int = 0,
    ) -> None:
        """Store the fully validated runtime components."""
        if batch_sampler_mode == external_step_mode:
            raise ValueError("Provide exactly one step source: batch_sampler or external_step_reader.")
        if metadata_mode and not batch_sampler_mode:
            raise ValueError("Metadata mode requires batch_sampler to define step/sample boundaries.")
        _validate_loader_components(
            topology=topology,
            dataset_reader_ranks=dataset_reader_ranks,
            dataset_reader=dataset_reader,
            metadata_reader=metadata_reader,
            direct_sample_loader=direct_sample_loader,
            metadata_mode=metadata_mode,
            double_buffer=double_buffer,
        )
        self._topology = topology
        self._dataset_reader_ranks = dataset_reader_ranks
        self._dataset_reader = dataset_reader
        self._metadata_reader = metadata_reader
        self._direct_sample_loader = direct_sample_loader
        self._metadata_mode = metadata_mode
        self._planner = planner
        self._data_constructor = data_constructor
        self._data_plane = data_plane
        self._model_transport = model_transport
        self._double_buffer = double_buffer
        self._config_fingerprint = config_fingerprint
        self._batch_sampler_mode = batch_sampler_mode
        self._external_step_mode = external_step_mode
        self._epoch = initial_epoch
        self._step = 0
        self._stopped = False
        self._last_plan_id: str | None = None
        self._last_plan: DistributedPackingPlan | None = None
        self._pending_local_keys: set[SampleKey] = set()
        self._pending_plan: DistributedPackingPlan | None = None
        self._prefetch_thread: Thread | None = None
        self._prefetch_condition = Condition()
        # ``None`` is a valid result (it signals end of stream), so a unique
        # sentinel distinguishes "not ready" from a completed empty batch.
        self._prefetch_result: Any = _PREFETCH_PENDING
        self._prefetch_error: BaseException | None = None
        self._prefetch_requested = False
        self._prefetch_in_flight = False
        self._prefetch_stream: Any = None

    def __iter__(self) -> "DistributedDataLoader":
        """Return this stateful distributed iterator."""
        return self

    def __next__(self) -> Any:
        """Collectively construct and return the next rank-local batch."""
        if self._double_buffer:
            received = self._broadcast_batch(self._take_prefetched())
        else:
            received = self._prepare_and_broadcast_batch()
        data = self._consume_batch(received)
        if self._double_buffer:
            self._start_prefetch()
        return data

    def _prepare_and_broadcast_batch(self) -> Any:
        """Prepare the next batch and broadcast it to model-parallel peers."""
        constructed_batch = None
        if self._data_plane.is_member:
            constructed_batch = self._produce_on_data_plane()
        return self._model_transport.broadcast(constructed_batch)

    def _prepare_next_batch(self) -> Any:
        """Select, balance, route, and construct samples without model-group broadcast."""
        if self._data_plane.is_member:
            return self._produce_on_data_plane()
        return None

    def _broadcast_batch(self, constructed_batch: Any) -> Any:
        """Broadcast the constructed batch to model-parallel peers on the caller thread."""
        return self._model_transport.broadcast(constructed_batch)

    def _consume_batch(self, received: Any) -> Any:
        """Validate the batch, commit Reader progress, and return its training data."""
        if received is None:
            self._stopped = True
            self._pending_local_keys.clear()
            self._pending_plan = None
            raise StopIteration
        if self._data_plane.is_member and self._pending_plan is None:
            raise ValueError("Received an active batch without a pending distributed packing plan.")

        # Dataset Reader buffers are committed only after construction and broadcast
        # have both succeeded, leaving checkpoint boundaries unambiguous.
        planning_reader = self._planning_reader()
        if planning_reader is not None:
            planning_reader.commit(self._pending_local_keys)
        self._pending_local_keys.clear()
        if self._pending_plan is not None:
            self._last_plan_id = self._pending_plan.plan_id
            self._last_plan = self._pending_plan
        self._pending_plan = None
        self._stopped = False
        self._step += 1
        return received

    def _start_prefetch(self) -> None:
        """Submit one background transaction to the persistent prefetch thread."""
        with self._prefetch_condition:
            if self._prefetch_in_flight:
                raise ValueError("A distributed local-batch prefetch is already in flight.")
            if self._prefetch_thread is None:
                self._prefetch_thread = Thread(
                    target=self._run_prefetch,
                    name=f"hp-data-prefetch-rank-{self._topology.global_rank}",
                    daemon=True,
                )
                self._prefetch_thread.start()
            self._prefetch_result = _PREFETCH_PENDING
            self._prefetch_error = None
            self._prefetch_requested = True
            self._prefetch_in_flight = True
            self._prefetch_condition.notify()

    def _run_prefetch(self) -> None:
        """Run submitted transactions on a persistent daemon thread."""
        # Accelerator device context is thread-local on CUDA/NPU. Bind it once
        # on the persistent worker instead of repeating it for every step.
        device_bound = False
        while True:
            with self._prefetch_condition:
                while not self._prefetch_requested:
                    self._prefetch_condition.wait()
                self._prefetch_requested = False
            try:
                if not device_bound:
                    self._bind_prefetch_device()
                    device_bound = True
                with self._prefetch_stream_context():
                    batch = self._prepare_next_batch()
            except BaseException as exc:  # The foreground re-raises failures at the next iterator boundary.
                with self._prefetch_condition:
                    self._prefetch_error = exc
                    self._prefetch_condition.notify_all()
                continue
            with self._prefetch_condition:
                self._prefetch_result = batch
                self._prefetch_condition.notify_all()

    def _bind_prefetch_device(self) -> None:
        """Bind the rank-local accelerator for collectives in the prefetch thread."""
        device = getattr(self._data_plane, "communication_device", None)
        if device is None:
            return
        device_module = getattr(torch, device.type, None)
        if device_module is not None and hasattr(device_module, "set_device"):
            device_module.set_device(device)

    def _prefetch_stream_context(self) -> Any:
        """Keep payload HCCL dependencies off the model's caller stream."""
        device = getattr(self._data_plane, "communication_device", None)
        if device is None or device.type != "npu":
            return nullcontext()
        device_module = getattr(torch, device.type)
        if self._prefetch_stream is None:
            self._prefetch_stream = device_module.Stream(device=device)
        # H2D, collective, wait and D2H must share this stream; isolating only
        # the collective omits the send buffer's producer dependency.
        return device_module.stream(self._prefetch_stream)

    def _take_prefetched(self) -> Any:
        """Wait for and return the current background result."""
        if not self._prefetch_in_flight:
            self._start_prefetch()
        with self._prefetch_condition:
            while self._prefetch_result is _PREFETCH_PENDING and self._prefetch_error is None:
                self._prefetch_condition.wait()
            result = self._prefetch_result
            error = self._prefetch_error
            self._prefetch_result = _PREFETCH_PENDING
            self._prefetch_error = None
            self._prefetch_in_flight = False
            self._prefetch_condition.notify_all()
        if error is not None:
            raise error
        if result is _PREFETCH_PENDING:
            raise ValueError("Double buffering completed without a finished prefetch result.")
        return result

    @property
    def last_plan_id(self) -> str | None:
        """Return the deterministic plan identifier of the last returned batch."""
        return self._last_plan_id

    def wait_for_prefetch(self) -> None:
        """Wait for in-flight data communication before training tears down groups.

        All ranks must call this at the same consumed-batch boundary. The
        prepared result remains available to the next iterator call.
        """
        with self._prefetch_condition:
            while (
                    self._prefetch_in_flight
                    and self._prefetch_result is _PREFETCH_PENDING
                    and self._prefetch_error is None
            ):
                self._prefetch_condition.wait()

    @property
    def last_plan(self) -> DistributedPackingPlan | None:
        """Return the last plan on data-plane ranks, otherwise ``None``."""
        return self._last_plan

    def state_dict(self) -> dict[str, Any]:
        """Return rank-local state at a completed distributed-batch boundary."""
        restart_prefetch = self._prefetch_in_flight
        if restart_prefetch:
            self._take_prefetched()
            # Prefetch is speculative until the trainer requests the batch.
            # Keep Reader buffers uncommitted so the checkpoint can replan it.
            self._pending_local_keys.clear()
            self._pending_plan = None
        if self._pending_local_keys:
            raise ValueError("Cannot checkpoint while a distributed batch is in flight.")
        state = {
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
            or self._prefetch_in_flight
        )
        if active_state:
            raise ValueError("load_state_dict must run before distributed iteration starts.")

    def _validate_checkpoint_identity(self, state: Mapping[str, Any]) -> None:
        expected = {
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
        if self._prefetch_in_flight:
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

    def _produce_on_data_plane(self) -> Any:
        plan = self._next_plan_control()
        if plan is None:
            return None

        if plan.step != self._step:
            raise ValueError(
                f"Planner returned step {plan.step}, but this rank expects step {self._step}."
            )
        self._pending_plan = plan
        selected_keys = set(plan.selected_keys)
        local_selected_keys = {
            key for key in selected_keys if key.reader_rank == self._topology.global_rank
        }
        self._pending_local_keys = local_selected_keys
        if self._metadata_mode:
            return self._produce_metadata_batch(plan)

        # VeOmni's external reader has already selected and packed this exact
        # step.  Reuse it when balancing kept every bin on its canonical owner;
        # moved steps continue through the normal payload exchange.
        canonical_match = False
        if self._external_step_mode and self._topology.is_constructor:
            reader = self._dataset_reader
            local_batch_plan = plan.local_batch_for(self._topology.data_rank)
            matches = getattr(reader, "canonical_plan_matches", None)
            canonical_match = callable(matches) and matches(local_batch_plan, self._topology.data_rank)
        if self._external_step_mode and self._data_plane.is_member:
            canonical_match = self._data_plane.all_ranks_true(canonical_match)
        if canonical_match and self._topology.is_constructor:
            canonical_batch = getattr(self._dataset_reader, "canonical_batch", None)
            if callable(canonical_batch):
                local_batch = canonical_batch()
                return self._require_batch(local_batch)

        outgoing = self._prepare_outgoing(plan, local_selected_keys)
        received_payloads = self._data_plane.exchange_prepared(outgoing)
        return self._construct_received_payloads(plan, received_payloads)

    def _produce_metadata_batch(self, plan: DistributedPackingPlan) -> Any:
        """Directly read constructor-assigned shared indices without payload A2A."""
        received_payloads: dict[SampleKey, Any] = {}
        if self._topology.is_constructor:
            if self._direct_sample_loader is None:
                raise ValueError("A metadata Data Constructor has no plan-aware sample loader.")
            received_payloads = self._direct_sample_loader.fetch(
                plan.local_sample_keys(self._topology.data_rank)
            )
        return self._construct_received_payloads(plan, received_payloads)

    def _construct_received_payloads(
            self,
            plan: DistributedPackingPlan,
            received_payloads: dict[SampleKey, Any],
    ) -> Any:
        """Construct one local batch and return it on the Data Constructor rank."""
        local_batch = None
        if self._topology.is_constructor:
            local_batch_plan = plan.local_batch_for(self._topology.data_rank)
            local_batch = self._data_constructor.construct(local_batch_plan, received_payloads)
        elif received_payloads:
            raise ValueError(
                f"Non-constructor rank {self._topology.global_rank} received unexpected sample payloads."
            )
        if not self._topology.is_constructor:
            return None
        return self._require_batch(local_batch)

    @staticmethod
    def _require_batch(local_batch: Any) -> Any:
        """Reject ``None`` because it is reserved for distributed end-of-stream."""
        if local_batch is None:
            raise ValueError("The Data Constructor must return a non-None batch.")
        return local_batch

    def _next_plan_control(self) -> DistributedPackingPlan | None:
        """Plan exactly one source-selected step, or return None at end of stream."""
        local_snapshot = self._fill_local_reader()
        snapshots = self._data_plane.gather_object_to_planner(local_snapshot)
        plan = None
        if self._topology.global_rank == self._data_plane.planner_rank:
            if snapshots is None:
                raise RuntimeError("Planner did not receive Dataset Reader snapshots.")
            plan = self._build_plan_control(snapshots)
        return self._data_plane.broadcast_from_planner(plan)

    def _fill_local_reader(self) -> _ReaderSnapshot:
        is_reader = self._topology.global_rank in self._dataset_reader_ranks
        if not is_reader:
            return _ReaderSnapshot(
                rank=self._topology.global_rank,
                exhausted=True,
                metadata=(),
                original_metadatas=(),
            )
        planning_reader = self._planning_reader()
        if planning_reader is None:
            raise ValueError(f"Dataset Reader rank {self._topology.global_rank} did not provide its reader.")

        # Step sources own membership; token targets must not pull a future step.
        planning_reader.prepare_next_step()
        original_metadatas = getattr(planning_reader, "original_metadatas", None)
        if original_metadatas is None:
            original_metadatas = getattr(planning_reader, "reference_bins", ())
        return _ReaderSnapshot(
            rank=self._topology.global_rank,
            exhausted=planning_reader.exhausted,
            metadata=planning_reader.metadata(),
            original_metadatas=tuple(original_metadatas),
        )

    def _build_plan_control(self, snapshots: tuple[Any, ...]) -> DistributedPackingPlan | None:
        """Return the current step's plan; None means end of stream."""
        normalized = self._normalize_reader_snapshots(snapshots)
        return self._plan_reader_snapshots(normalized)

    def _normalize_reader_snapshots(
            self,
            snapshots: tuple[Any, ...],
    ) -> tuple[_ReaderSnapshot, ...]:
        normalized = []
        for snapshot in snapshots:
            if not isinstance(snapshot, _ReaderSnapshot):
                raise ValueError("A data-plane rank contributed an invalid reader snapshot.")
            if snapshot.rank not in self._data_plane.ranks:
                raise ValueError(f"Rank {snapshot.rank} is outside the configured data plane.")
            normalized.append(snapshot)
        contributed_ranks = [snapshot.rank for snapshot in normalized]
        if len(contributed_ranks) != len(set(contributed_ranks)) or set(contributed_ranks) != set(
                self._data_plane.ranks
        ):
            raise ValueError(
                f"Data-plane reader snapshots have invalid rank coverage {contributed_ranks}."
            )
        return tuple(normalized)

    def _plan_reader_snapshots(
            self, normalized: tuple[_ReaderSnapshot, ...],
    ) -> DistributedPackingPlan | None:
        reader_snapshots = [
            snapshot for snapshot in normalized if snapshot.rank in self._dataset_reader_ranks
        ]
        if self._batch_sampler_mode:
            selection = self._select_native_batch(reader_snapshots)
        else:
            selection = self._select_external_step(reader_snapshots)
        return None if selection is None else self._planner.plan(selection, step=self._step)

    def _select_external_step(self, snapshots: list[_ReaderSnapshot]) -> StepSampleSelection | None:
        """Freeze the exact sample set emitted by an external legacy producer.

        Each Dataset Reader has already run its local VeOmni selector for the
        current step.  We preserve that union and let the HP planner only
        change target-rank placement.  The original metadata groups are used to
        validate that every local producer emitted the expected number of
        packs; the planner is still free to repack the frozen samples.
        """
        reader_snapshots = snapshots
        if not reader_snapshots:
            return None
        if all(snapshot.exhausted for snapshot in reader_snapshots):
            return None
        if any(snapshot.exhausted for snapshot in reader_snapshots):
            raise ValueError("External-step readers exhausted at different forward/backward steps.")
        expected_bins = self._planner.distributed_bin_count
        original_metadatas = tuple(
            packing_bin
            for snapshot in sorted(reader_snapshots, key=lambda item: item.rank)
            for packing_bin in snapshot.original_metadatas
        )
        if len(original_metadatas) != expected_bins:
            raise ValueError(
                f"External-step producers emitted {len(original_metadatas)} local packs, "
                f"expected {expected_bins}."
            )
        samples = tuple(
            item
            for snapshot in sorted(reader_snapshots, key=lambda item: item.rank)
            for item in snapshot.metadata
        )
        if not samples:
            raise ValueError("External-step producers emitted no samples for an active step.")
        # Reader-local streams have different sample counts, so their source
        # ordinals are not contiguous globally.  Renumber only the frozen
        # selection; SampleKey remains the routing identity.
        external_samples = tuple(
            BufferedSampleMetadata(item.key, item.metadata, position)
            for position, item in enumerate(samples)
        )
        return StepSampleSelection(
            samples=external_samples,
            reference_bins=tuple(
                tuple(item.key for item in packing_bin)
                for packing_bin in original_metadatas
            ),
        )

    def _select_native_batch(self, snapshots: list[_ReaderSnapshot]) -> StepSampleSelection | None:
        """Freeze this forward/backward round exactly as native DP samplers selected it."""
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
    ) -> PreparedPayloadExchange:
        outgoing: dict[int, list[tuple[SampleKey, Any]]] = {}
        if local_selected_keys:
            if self._dataset_reader is None:
                raise ValueError("A planned Dataset Reader rank has no Dataset Reader.")
            payloads = self._dataset_reader.selected_payloads(local_selected_keys)
            target_by_key = {
                key: self._topology.constructor_ranks[data_rank]
                for data_rank, local_batch in enumerate(plan.local_batches)
                for packing_bin in local_batch
                for key in packing_bin.sample_keys
            }
            for key, payload in payloads:
                outgoing.setdefault(target_by_key[key], []).append((key, payload))
        return self._data_plane.prepare_exchange(outgoing)

    def _planning_reader(self) -> Any | None:
        """Return this rank's online or metadata-only Dataset Reader."""
        if self._dataset_reader is not None:
            return self._dataset_reader
        return self._metadata_reader

__all__ = ["DistributedDataLoader"]
