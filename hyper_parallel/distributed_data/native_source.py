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
"""Process-owned native sampling, planning and complete-step checkpoints."""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Union

from hyper_parallel.data.parallel.batch_sampler import (
    get_sampler_checkpoint_state, has_torchdata_state, sampler_optimizer_step_progress,
)
from hyper_parallel.distributed_data.balance_logging import format_balance_stats, logger as balance_logger
from hyper_parallel.distributed_data.batch_sampler import _ReaderPlanInput
from hyper_parallel.distributed_data.schema import DistributedPackingPlan, SampleKey, SampleMetadata
from hyper_parallel.distributed_data.transport import DataPlaneTransport, PreparedPayloadExchange

_PlanControl = Union[DistributedPackingPlan, tuple[DistributedPackingPlan, ...], None]


@dataclass
class PreparedWindow:
    """Carry CPU batches and the checkpoint at the window's optimizer boundary."""

    batches: deque[Any]
    exhausted: bool
    checkpoint: dict[str, Any] | None
    balance_logs: list[str] = field(default_factory=list)


class NativeWindowSource:
    """Own speculative native reads without changing the delivered training cursor."""

    def __init__(self, *, topology: Any, dataset_reader_ranks: tuple[int, ...], dataset_reader: Any,
                 metadata_reader: Any, direct_sample_loader: Any, metadata_mode: bool, planner: Any,
                 data_constructor: Any, initial_epoch: int, initial_consumed_samples: int,
                 broadcast_plan: bool, log_balance: bool, accumulation_steps: int,
                 balance_window_size: int) -> None:
        """Capture CPU-only components before any dataset workers are started."""
        self._topology = topology
        self._dataset_reader_ranks = frozenset(dataset_reader_ranks)
        self._dataset_reader = dataset_reader
        self._metadata_reader = metadata_reader
        self._direct_sample_loader = direct_sample_loader
        self._metadata_mode = metadata_mode
        self._planner = planner
        self._data_constructor = data_constructor
        self._broadcast_plan = broadcast_plan
        self._log_balance = log_balance
        self._accumulation_steps = accumulation_steps
        self._balance_window_size = balance_window_size
        self._epoch = initial_epoch
        self._initial_consumed_samples = initial_consumed_samples
        self._step = 0
        self._stopped = False
        self._pending_local_keys = set()
        self._window_plans = deque()
        self._balance_logs = []
        self._data_plane = None

    @property
    def global_rank(self) -> int:
        """Return the logical sample-owner rank in the training mesh."""
        return self._topology.global_rank

    def bind_transport(self, transport: DataPlaneTransport) -> None:
        """Attach the child-owned data group after deserializing CPU components.

        Args:
            transport: Data-plane collectives owned by this source process.
        """
        self._data_plane = transport

    def prefetch(self) -> None:
        """In-process sources prepare their window when the caller requests the result."""

    @staticmethod
    def is_ready(timeout: float = 0.0) -> bool:
        """An in-process source can enter its next synchronous read immediately.

        Args:
            timeout: Maximum seconds to wait for a ready result.
        """
        del timeout
        return True

    def __next__(self) -> PreparedWindow:
        """Prepare one window; only parent delivery commits its checkpoint."""
        return self._prepare_window()

    def close(self) -> None:
        """Release source DataLoader workers in their owning process."""
        reader = self._planning_reader()
        if reader is not None:
            reader.close()
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.close()

    def set_epoch(self, epoch: int) -> None:
        """Reset sampling after the parent has drained the previous epoch.

        Args:
            epoch: Non-negative epoch to select before the next source read.
        """
        for reader in (self._dataset_reader, self._metadata_reader, self._direct_sample_loader):
            if reader is not None:
                reader.set_epoch(epoch)
        self._epoch = epoch
        self._step = 0
        self._initial_consumed_samples = 0
        self._stopped = False
        self._pending_local_keys.clear()
        self._window_plans.clear()
        self._balance_logs = []

    def _prepare_batch(self) -> Any:
        """Construct one CPU round; model peers receive batches only at parent delivery."""
        if self._stopped:
            raise StopIteration
        batch = self._prepare_next_batch()
        if self._stopped:
            raise StopIteration
        return self._consume_batch(batch)

    def _prepare_next_batch(self) -> Any:
        if self._data_plane.is_member:
            return self._produce_on_data_plane()
        return None

    def _prepare_window(self) -> PreparedWindow:
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
        checkpoint = self.snapshot() if not exhausted and self._step % self._accumulation_steps == 0 else None
        return PreparedWindow(batches, exhausted, checkpoint, balance_logs=self._balance_logs)

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

    def snapshot(self) -> dict[str, Any]:
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
        self._validate_optimizer_boundary(state, sampler, step)
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

    def _validate_optimizer_boundary(self, state: Mapping[str, Any], sampler: Mapping[str, Any] | None,
                                     step: int) -> None:
        """Reject speculative or incomplete optimizer-step checkpoints."""
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
        self._validate_dataset_state(state, payload_state)
        if payload_state is not None:
            payload_state["epoch"] = epoch
        if reader is not None:
            reader_state.update(epoch=epoch, sampler=sampler_state)
            if reader_state["sample_loader"] is not None:
                reader_state["sample_loader"] = payload_state
            reader.load_state_dict(reader_state)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.load_state_dict(payload_state)

    @staticmethod
    def _validate_dataset_state(state: Mapping[str, Any], payload_state: Mapping[str, Any] | None) -> None:
        """Validate balanced-reader metadata without interpreting user Dataset state."""
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
