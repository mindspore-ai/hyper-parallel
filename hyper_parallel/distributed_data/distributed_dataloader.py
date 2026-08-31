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
"""Synchronous Dataset Reader, Planner, and Data Constructor orchestration."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from hyper_parallel.distributed_data.data_constructor import PackingDataConstructor
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    ConstructedBatch,
    DistributedPackingPlan,
    SampleKey,
)
from hyper_parallel.distributed_data.sidecar import PlannedSampleLoader, SidecarMetadataReader
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


@dataclass(frozen=True)
class _PlanControl:
    kind: _ControlKind
    plan: DistributedPackingPlan | None = None
    error: str | None = None


class DistributedDataLoader(Iterator[Any]):
    """Yield dynamically packed local batches on every training rank.

    One call to :func:`next` is a synchronous transaction: Dataset Readers fill
    their buffers, the Planner assigns raw samples, payloads move over the CPU
    data plane, Data Constructors pack/collate, and the result is broadcast to
    the model-parallel peers for the same DP coordinate.
    """

    VERSION = 2

    def __init__(
            self,
            *,
            topology: DataTopology,
            dataset_reader_ranks: tuple[int, ...],
            dataset_reader: DatasetReader | None,
            sidecar_reader: SidecarMetadataReader | None,
            direct_sample_loader: PlannedSampleLoader | None,
            sidecar_mode: bool,
            planner: DynamicPackingPlanner,
            data_constructor: PackingDataConstructor,
            data_plane: DataPlaneTransport,
            model_transport: ModelParallelTransport,
            buffer_size_multiplier: float,
            max_buffered_samples: int,
            config_fingerprint: str,
    ) -> None:
        """Store the fully validated runtime components."""
        if not isinstance(sidecar_mode, bool):
            raise ValueError("sidecar_mode must be boolean.")
        if sidecar_mode and dataset_reader is not None:
            raise ValueError("Sidecar mode must not configure an online Dataset Reader.")
        if not sidecar_mode and (sidecar_reader is not None or direct_sample_loader is not None):
            raise ValueError("Online mode must not configure sidecar loading components.")
        is_reader = topology.global_rank in dataset_reader_ranks
        planning_reader = sidecar_reader if sidecar_mode else dataset_reader
        if is_reader != (planning_reader is not None):
            raise ValueError("Dataset Reader ownership does not match dataset_reader_ranks.")
        if sidecar_mode and topology.is_constructor != (direct_sample_loader is not None):
            raise ValueError("Every sidecar Data Constructor must own one plan-aware sample loader.")
        self._topology = topology
        self._dataset_reader_ranks = dataset_reader_ranks
        self._dataset_reader = dataset_reader
        self._sidecar_reader = sidecar_reader
        self._direct_sample_loader = direct_sample_loader
        self._sidecar_mode = sidecar_mode
        self._planner = planner
        self._data_constructor = data_constructor
        self._data_plane = data_plane
        self._model_transport = model_transport
        self._buffer_size_multiplier = buffer_size_multiplier
        self._max_buffered_samples = max_buffered_samples
        self._config_fingerprint = config_fingerprint
        self._epoch = 0
        self._step = 0
        self._stopped = False
        self._last_plan_id: str | None = None
        self._last_plan: DistributedPackingPlan | None = None
        self._pending_local_keys: set[SampleKey] = set()

    def __iter__(self) -> "DistributedDataLoader":
        """Return this stateful distributed iterator."""
        return self

    def __next__(self) -> Any:
        """Collectively construct and return the next rank-local batch."""
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
        if received.error is not None:
            raise RuntimeError(received.error)
        if received.step != self._step:
            raise ValueError(
                f"Constructed batch step mismatch: expected {self._step}, got {received.step}."
            )
        if received.stopped:
            self._stopped = True
            raise StopIteration
        if was_stopped:
            raise ValueError("Distributed DataLoader stopped state differs across model-parallel peers.")
        if not received.plan_id:
            raise ValueError("An active constructed batch must include a plan_id.")

        # Dataset Reader buffers are committed only after construction and delivery
        # have both succeeded, leaving checkpoint boundaries unambiguous.
        planning_reader = self._planning_reader()
        if planning_reader is not None:
            planning_reader.commit(self._pending_local_keys)
        self._pending_local_keys.clear()
        self._last_plan_id = received.plan_id
        self._stopped = False
        self._step += 1
        return received.data

    @property
    def last_plan_id(self) -> str | None:
        """Return the most recently delivered deterministic plan identifier."""
        return self._last_plan_id

    @property
    def last_plan(self) -> DistributedPackingPlan | None:
        """Return the last plan on data-plane ranks, otherwise ``None``."""
        return self._last_plan

    def state_dict(self) -> dict[str, Any]:
        """Return rank-local state at a completed distributed-batch boundary."""
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
            "sidecar_reader": self._sidecar_reader.state_dict() if self._sidecar_reader is not None else None,
            "direct_sample_loader": (
                self._direct_sample_loader.state_dict() if self._direct_sample_loader is not None else None
            ),
        }
        try:
            return copy.deepcopy(state)
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not checkpointable: {exc}") from exc

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a fixed-topology rank-local checkpoint.

        Args:
            state_dict: State produced on this same global rank.
        """
        if self._step != 0 or self._pending_local_keys:
            raise ValueError("load_state_dict must run before distributed iteration starts.")
        try:
            state = copy.deepcopy(dict(state_dict))
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not copyable: {exc}") from exc
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
        reader_state = state.get("dataset_reader")
        sidecar_reader_state = state.get("sidecar_reader")
        direct_sample_state = state.get("direct_sample_loader")
        if (self._dataset_reader is None) != (reader_state is None):
            raise ValueError("Distributed DataLoader checkpoint Dataset Reader ownership changed.")
        if (self._sidecar_reader is None) != (sidecar_reader_state is None):
            raise ValueError("Distributed DataLoader checkpoint sidecar ownership changed.")
        if (self._direct_sample_loader is None) != (direct_sample_state is None):
            raise ValueError("Distributed DataLoader checkpoint direct-reader ownership changed.")
        for component_name, component_state in (
                ("Dataset Reader", reader_state),
                ("sidecar reader", sidecar_reader_state),
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
        if self._dataset_reader is not None:
            self._dataset_reader.load_state_dict(reader_state)
        if self._sidecar_reader is not None:
            self._sidecar_reader.load_state_dict(sidecar_reader_state)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.load_state_dict(direct_sample_state)
        self._epoch = epoch
        self._step = step
        self._stopped = stopped
        self._last_plan_id = last_plan_id

    def set_epoch(self, epoch: int) -> None:
        """Reset an exhausted loader for a deterministic new Dataset epoch.

        Args:
            epoch: Non-negative Dataset epoch.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, but got {epoch!r}.")
        if self._step != 0 and not self._stopped:
            raise ValueError("set_epoch requires a fresh or exhausted Distributed DataLoader.")
        if self._dataset_reader is not None:
            self._dataset_reader.set_epoch(epoch)
        if self._sidecar_reader is not None:
            self._sidecar_reader.set_epoch(epoch)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.set_epoch(epoch)
        self._epoch = epoch
        self._step = 0
        self._stopped = False
        self._last_plan_id = None
        self._last_plan = None

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
        self._last_plan = plan
        selected_keys = set(plan.selected_keys)
        local_selected_keys = {
            key for key in selected_keys if key.reader_rank == self._topology.global_rank
        }
        self._pending_local_keys = local_selected_keys
        if self._sidecar_mode:
            return self._produce_sidecar_batch(plan)

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

    def _produce_sidecar_batch(self, plan: DistributedPackingPlan) -> ConstructedBatch | None:
        """Directly read constructor-assigned indices without payload A2A."""
        received_payloads: dict[SampleKey, Any] = {}
        fetch_error = None
        if self._topology.is_constructor:
            try:
                if self._direct_sample_loader is None:
                    raise ValueError("A sidecar Data Constructor has no plan-aware sample loader.")
                constructor_plan = plan.constructor_for(self._topology.data_rank)
                received_payloads = self._direct_sample_loader.fetch(constructor_plan)
            except Exception as exc:
                fetch_error = self._format_error("sidecar direct read", exc)
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
        )

    def _build_plan_control(self, snapshots: tuple[Any, ...]) -> _PlanControl:
        normalized = []
        for snapshot in snapshots:
            if not isinstance(snapshot, _ReaderSnapshot):
                return _PlanControl("error", error="A data-plane rank contributed an invalid reader snapshot.")
            expected_reader = snapshot.rank in self._dataset_reader_ranks
            if snapshot.is_reader != expected_reader:
                return _PlanControl(
                    "error",
                    error=f"Rank {snapshot.rank} reported inconsistent Dataset Reader ownership.",
                )
            normalized.append(snapshot)
        contributed_ranks = [snapshot.rank for snapshot in normalized]
        if len(contributed_ranks) != len(set(contributed_ranks)) or set(contributed_ranks) != set(
                self._data_plane.ranks
        ):
            return _PlanControl(
                "error",
                error=f"Data-plane reader snapshots have invalid rank coverage {contributed_ranks}.",
            )
        errors = sorted((snapshot.rank, snapshot.error) for snapshot in normalized if snapshot.error is not None)
        if errors:
            return _PlanControl("error", error=errors[0][1])

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

        reader_snapshots = [snapshot for snapshot in normalized if snapshot.is_reader]
        candidates = tuple(metadata for snapshot in reader_snapshots for metadata in snapshot.metadata)
        try:
            plan = self._planner.plan(candidates, step=self._step)
        except Exception as exc:
            return _PlanControl("error", error=self._format_error("Planner", exc))
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
                f"the Planner found {self._planner.distributed_bin_count} samples."
            ),
        )

    def _prepare_outgoing(
            self,
            plan: DistributedPackingPlan,
            local_selected_keys: set[SampleKey],
    ) -> tuple[PreparedPayloadExchange | None, str | None]:
        outgoing: dict[int, list[tuple[SampleKey, Any]]] = {}
        try:
            if local_selected_keys:
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

    def _planning_reader(self) -> DatasetReader | SidecarMetadataReader | None:
        """Return this rank's online or metadata-only Dataset Reader."""
        if self._dataset_reader is not None:
            return self._dataset_reader
        return self._sidecar_reader

    def _constructor_envelope(
            self,
            *,
            stopped: bool = False,
            error: str | None = None,
    ) -> ConstructedBatch | None:
        self._pending_local_keys.clear()
        if not self._topology.is_constructor:
            return None
        return ConstructedBatch(step=self._step, plan_id=None, stopped=stopped, error=error)

    @staticmethod
    def _format_error(stage: str, exception: Exception) -> str:
        return f"{stage} failed: {type(exception).__name__}: {exception}"


__all__ = ["DistributedDataLoader"]
