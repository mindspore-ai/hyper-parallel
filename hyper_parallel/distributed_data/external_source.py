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
"""Process-owned preparation of externally selected raw-sample steps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import chain
from typing import Any

from hyper_parallel.distributed_data.schema import DistributedPackingPlan, SampleKey, SampleMetadata, WorkloadCost
from hyper_parallel.distributed_data.transport import DataPlaneTransport, PayloadExchangeWork


@dataclass(frozen=True)
class LocalBatch:
    """Retain a collated CPU step and its balancing statistics."""

    data: Any
    stats: dict[str, Any] | None = None
    host_batch: Any = None


@dataclass(frozen=True)
class _PendingBatch:
    """A planned step whose payload exchange may still be in flight."""

    plan: DistributedPackingPlan
    stats: dict[str, Any]
    retained_payloads: dict[SampleKey, Any]
    payload_work: PayloadExchangeWork | None


class ExternalStepProducer:
    """Keep source iteration, planning, collation and communication in one process."""

    def __init__(self, source: Any, *, config: Any, metadata_fn: Any, pack_fn: Any,
                 collate_fn: Any, planner: Any, global_rank: int, ranks: tuple[int, ...]) -> None:
        """Capture an unstarted external source and its CPU callbacks."""
        self._source = source
        self.config = config
        self._metadata_fn = metadata_fn
        self._pack_fn = pack_fn
        self._collate_fn = collate_fn
        self._planner = planner
        self._global_rank = global_rank
        self.group_ranks = ranks
        self._data_rank = ranks.index(global_rank)
        self._transport = None
        self._iterator = None
        self._step = 0

    @property
    def global_rank(self) -> int:
        """Return the logical source rank in the training mesh."""
        return self._global_rank

    def bind_transport(self, transport: DataPlaneTransport) -> None:
        """Bind collectives reconstructed in the owning process.

        Args:
            transport: Data-plane collectives owned by this source process.
        """
        self._transport = transport

    def __next__(self) -> LocalBatch:
        """Construct one selected step without changing sample membership."""
        if self._iterator is None:
            self._iterator = iter(self._source)
        result = self._construct_batch(next(self._iterator), self._step)
        self._step += 1
        return result

    def reset_iteration(self) -> None:
        """Discard the exhausted iterator while retaining the original source object."""
        self._iterator = None
        self._step = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the external source epoch before constructing its next iterator.

        Args:
            epoch: Non-negative epoch to select before the next source read.
        """
        self.reset_iteration()
        setter = getattr(self._source, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def close(self) -> None:
        """Drop source iterators in the process that owns their workers."""
        self._iterator = None

    def _construct_batch(
            self,
            raw_bins: Sequence[Sequence[Any]],
            step: int,
    ) -> LocalBatch:
        metadata, local_payloads = self._read_step(raw_bins, step)
        gathered = self._transport.all_gather_object(metadata)
        plan, stats = self._plan_step(gathered, step)
        return self._finish_batch(self._begin_batch(local_payloads, plan, stats))

    def _begin_batch(
            self, local_payloads: dict[SampleKey, Any], plan: DistributedPackingPlan, stats: dict[str, Any],
    ) -> _PendingBatch:
        """Start payload exchange for moved samples while retaining local destinations."""
        if stats["moved_samples"]:
            outgoing: dict[int, list[tuple[SampleKey, Any]]] = {}
            retained_payloads = {}
            for data_rank, local_batch in enumerate(plan.local_batches):
                target_rank = self.group_ranks[data_rank]
                owned = [
                    (key, local_payloads[key])
                    for packing_bin in local_batch
                    for key in packing_bin.sample_keys
                    if key.reader_rank == self._global_rank
                ]
                if target_rank == self._global_rank:
                    retained_payloads.update(owned)
                else:
                    outgoing[target_rank] = owned
            prepared = self._transport.prepare_exchange(outgoing)
            work = self._transport.begin_exchange_prepared(prepared)
        else:
            # Every rank sees the same plan and skips an unchanged exchange.
            retained_payloads = local_payloads
            work = None
        return _PendingBatch(plan, stats, retained_payloads, work)

    def _finish_batch(self, pending: _PendingBatch) -> LocalBatch:
        """Complete payload exchange, pack and collate the local batch, return CPU tensors."""
        received = pending.retained_payloads
        if pending.payload_work is not None:
            received.update(pending.payload_work.wait())
        target = pending.plan.local_batch_for(self._data_rank)
        host_batch = self._collate_fn([
            self._pack_fn([received[key] for key in packing_bin.sample_keys], self.config.seq_len)
            for packing_bin in target
        ])
        return LocalBatch(host_batch, pending.stats, host_batch)

    def _read_step(self, raw_bins: Sequence[Sequence[Any]], step: int) -> tuple[tuple[Any, ...], dict[SampleKey, Any]]:
        """Bind raw source bins to keyed metadata and cached payloads for one step."""
        if len(raw_bins) != self.config.local_batch_size or any(not raw_bin for raw_bin in raw_bins):
            raise ValueError("Each source step must contain local_batch_size non-empty raw-sample bins.")
        key_bins = []
        entries = []
        payloads = {}
        for raw_bin in raw_bins:
            keys = []
            for sample in raw_bin:
                item_metadata = self._metadata_fn(sample)
                if not isinstance(item_metadata, SampleMetadata):
                    raise ValueError("metadata_fn must return SampleMetadata.")
                key = SampleKey(self._global_rank, step, len(entries))
                entries.append(replace(item_metadata, key=key))
                payloads[key] = sample
                keys.append(key)
            key_bins.append(tuple(keys))
        return (tuple(key_bins), tuple(entries)), payloads

    def _plan_step(self, gathered: Sequence[Any], step: int) -> tuple[DistributedPackingPlan, dict[str, Any]]:
        planned = self._make_plan(gathered, step)
        return self._transport.broadcast_from_planner(planned) if self.config.broadcast_plan else planned

    def _make_plan(self, gathered: Sequence[Any], step: int) -> Any:
        """Build a plan and balancing statistics on the designated planning ranks."""
        if self.config.broadcast_plan and self._global_rank != self._transport.planner_rank:
            return None
        rank_bins, rank_entries = zip(*gathered)
        metadata = {item.key: item for item in chain.from_iterable(rank_entries)}
        if self.config.balance_window_size == 1:
            plan = self._planner.plan(
                tuple(metadata.values()), reference_bins=list(chain.from_iterable(rank_bins)), step=step,
            )
            costs, decision = self._planner.last_sample_costs, self._planner.last_balance_decision
        else:
            plan, costs, decision = self._plan_windows(rank_bins, metadata, step)
        bin_stats = self._bin_statistics(rank_bins, metadata, plan, costs)
        stats = self._statistics(plan, step, bin_stats["bins_before"])
        stats.update(decision)
        if self._transport.planner_rank == 0:
            stats.update(bin_stats)
        return plan, stats

    def _plan_windows(
            self, rank_bins: Sequence[Sequence[Sequence[SampleKey]]],
            metadata: dict[SampleKey, SampleMetadata], step: int,
    ) -> tuple[DistributedPackingPlan, dict[SampleKey, WorkloadCost], dict[str, Any]]:
        """Plan consecutive bins inside this source yield, retaining one physical destination per rank."""
        rounds = list(zip(*rank_bins, strict=True))
        plans = []
        windows = []
        costs = {}
        width = self.config.balance_window_size
        for start in range(0, len(rounds), width):
            bins = list(chain.from_iterable(rounds[start:start + width]))
            selected = [metadata[key] for packing_bin in bins for key in packing_bin]
            plans.extend(self._planner.plan_window(selected, reference_bins=bins, step=step * len(rounds) + start))
            costs.update(self._planner.last_sample_costs)
            windows.append({
                **self._planner.last_balance_decision, "first_micro_batch": start + 1, "last_micro_batch": len(plans),
            })

        ranks = range(len(rank_bins))
        local_batches = tuple(tuple(plan.local_batch_for(rank)[0] for plan in plans) for rank in ranks)
        rank_costs = tuple(sum((plan.rank_costs[rank] for plan in plans), WorkloadCost()) for rank in ranks)
        combined = DistributedPackingPlan(step, self.config.seq_len, local_batches, rank_costs, validate=False)
        original = math.fsum(window["original_objective"] for window in windows)
        candidate = math.fsum(window["candidate_objective"] for window in windows)
        return combined, costs, {
            "windows": windows, "original_objective": original, "candidate_objective": candidate,
            "accepted_objective": math.fsum(window["accepted_objective"] for window in windows),
            "relative_gain": (original - candidate) / original if original else 0.0,
            "min_balance_gain": self.config.min_balance_gain, "accepted": any(window["accepted"] for window in windows),
        }

    def _bin_statistics(
            self, rank_bins: Sequence[Sequence[Sequence[SampleKey]]], metadata_by_key: dict[SampleKey, SampleMetadata],
            plan: DistributedPackingPlan, sample_costs: dict[SampleKey, WorkloadCost],
    ) -> dict[str, Any]:
        """Summarize original and accepted bins using the planner's estimates, including padding."""

        def _describe(keys: Sequence[SampleKey]) -> dict[str, Any]:
            tokens = sum(metadata_by_key[key].pack_tokens for key in keys)
            padding = max(0, self._planner.seq_len - tokens)
            return {
                "samples": len(keys), "seq_len": tokens, "padding": padding,
                "cost": math.fsum(sample_costs[key].llm for key in keys) + self._planner.padding_cost(padding).llm,
            }

        before = tuple(tuple(_describe(keys) for keys in key_bins) for key_bins in rank_bins)
        after = tuple(
            tuple(_describe(packing_bin.sample_keys) for packing_bin in local_batch)
            for local_batch in plan.local_batches
        )
        return {"bins_before": before, "bins_after": after}

    def _statistics(
            self, plan: DistributedPackingPlan, step: int, bins_before: Sequence[Sequence[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Count cross-rank raw-sample transfers, excluding locally retained samples."""
        before = [math.fsum(packing_bin["cost"] for packing_bin in bins) for bins in bins_before]
        after = [cost.llm for cost in plan.rank_costs]
        send_samples = [0] * len(self.group_ranks)
        recv_samples = [0] * len(self.group_ranks)
        moved = 0
        source_indices = {rank: index for index, rank in enumerate(self.group_ranks)}
        for data_rank, local_batch in enumerate(plan.local_batches):
            for packing_bin in local_batch:
                for key in packing_bin.sample_keys:
                    source_index = source_indices[key.reader_rank]
                    if source_index != data_rank:
                        send_samples[source_index] += 1
                        recv_samples[data_rank] += 1
                        moved += 1
        mean_before = math.fsum(before) / len(before)
        mean_after = math.fsum(after) / len(after)
        max_before, max_after = max(before), max(after)
        return {
            "step": step,
            "scope": "node",
            "objective": self._planner.objective,
            "group_ranks": self.group_ranks,
            "cost_before": tuple(before),
            "cost_after": tuple(after),
            "max_cost_before": max_before,
            "max_cost_after": max_after,
            "relative_spread_before": (max_before - min(before)) / mean_before if mean_before else 0.0,
            "relative_spread_after": (max_after - min(after)) / mean_after if mean_after else 0.0,
            "moved_samples": moved,
            "send_samples": tuple(send_samples),
            "recv_samples": tuple(recv_samples),
        }
