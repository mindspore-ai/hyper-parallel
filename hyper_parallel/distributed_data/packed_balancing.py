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
"""Rebalance raw samples from already-selected rank-local packing batches."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from threading import Thread
from typing import TYPE_CHECKING, Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.balance_logging import log_balance_stats
from hyper_parallel.distributed_data.cost_model import CostModel, resolve_cost_model_id
from hyper_parallel.distributed_data.device_prefetch import DeviceStepPrefetcher
from hyper_parallel.distributed_data.locality import create_locality_groups
from hyper_parallel.distributed_data.planner import LPTPackingPlanner
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    PackingConstraints,
    SampleKey,
    SampleMetadata,
    StepSampleSelection,
)
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import DataPlaneTransport

if TYPE_CHECKING:
    from hyper_parallel.distributed_data.api import DistributedDatasetConfig


@dataclass(frozen=True)
class _LocalBatch:
    data: Any
    stats: dict[str, Any] | None = None


class _LocalBalancingIterator(Iterator[Any]):
    """Keep one fully balanced Host batch ahead of the training consumer."""

    def __init__(self, loader: LocalBalancingDataLoader) -> None:
        """Initialize a source iterator and one empty prefetch slot."""
        # Start source workers on the calling thread before entering prefetch.
        self._source = iter(loader.local_dataloader)
        self._loader = loader
        self._step = 0
        self.finished = False
        self._thread: Thread | None = None
        self._result: _LocalBatch | None = None
        self._error: BaseException | None = None
        self._stream = None

    def __next__(self) -> Any:
        """Deliver the current batch and prepare the following complete step."""
        if self.finished or self._limit_reached():
            self.finished = True
            raise StopIteration
        try:
            if self._loader.config.double_buffer:
                if self._thread is None:
                    self._start_prefetch()
                self.wait_for_prefetch()
                self._thread = None
                if self._error is not None:
                    raise self._error
                result = self._result
                self._result = None
                if result is None:
                    raise RuntimeError("Local balancing prefetch completed without a batch.")
            else:
                result = self._collect_batch()
        except BaseException:
            self.finished = True
            raise
        self._step += 1
        self._loader.last_balance_stats = result.stats
        if self._loader.config.double_buffer and not self._limit_reached():
            self._start_prefetch()
        return self._loader._deliver_batch(result, self._step)

    def _limit_reached(self) -> bool:
        return self._loader.max_steps is not None and self._step >= self._loader.max_steps

    def _collect_batch(self) -> _LocalBatch:
        return self._loader._construct_batch(next(self._source), self._step)

    def _start_prefetch(self) -> None:
        self._result = None
        self._error = None
        self._thread = Thread(
            target=self._run_prefetch,
            name="hp-local-balance-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _run_prefetch(self) -> None:
        try:
            transport = self._loader._transport
            device = transport.communication_device if transport is not None else None
            context = nullcontext()
            if device is not None and device.type in ("cuda", "npu"):
                accelerator = getattr(torch, device.type)
                accelerator.set_device(device)
                if self._stream is None:
                    self._stream = accelerator.Stream(device=device)
                context = accelerator.stream(self._stream)
            # Include the source read, metadata, planning, H2D, A2A, D2H and
            # CPU pack/collate in the same transaction and producer stream.
            with context:
                self._result = self._collect_batch()
        except BaseException as exc:
            self._error = exc

    def wait_for_prefetch(self) -> None:
        """Finish pending communication without consuming its prepared batch."""
        if self._thread is not None:
            self._thread.join()

    def close(self) -> None:
        """Drain the producer before resetting the source or its epoch."""
        self.wait_for_prefetch()
        self.finished = True
        self._thread = None
        self._result = None
        self._error = None


class LocalBalancingDataLoader:
    """Balance a frozen local-packer step using only one locality domain.

    Raw samples, not collated tensors or entire indivisible packs, are routed.
    The source controls sampling/filtering/packing and microbatch membership;
    this wrapper never reads future samples to improve a current step's score.
    All ranks must yield the same number of steps. Metadata and planner output
    are trusted; local failures propagate without a collective error handshake.
    """

    def __init__(
            self,
            local_dataloader: Iterable[Sequence[Sequence[Any]]],
            *,
            config: DistributedDatasetConfig,
            metadata_fn: Callable[[Any], SampleMetadata],
            pack_fn: Callable[[Sequence[Any], int], Any],
            collate_fn: Callable[[Sequence[Any]], Any],
            planner: Any,
            transport: DataPlaneTransport | None,
            global_rank: int,
            balancing_scope: str,
            enable_balancing: bool,
            bin_stats_fn: Callable[[Iterable[SampleMetadata]], dict[str, Any]] | None = None,
            device_prefetch: DeviceStepPrefetcher | None = None,
            max_steps: int | None = None,
            balance_stats_callback: Callable[[dict[str, Any], int, int | None], None] | None = None,
    ) -> None:
        """Store the source, planner, and locality-scoped data plane."""
        if max_steps is not None and (not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None.")
        self.local_dataloader = local_dataloader
        self.config = config
        self.group_ranks = transport.ranks if transport is not None else (global_rank,)
        self.balancing_scope = balancing_scope
        self.enable_balancing = enable_balancing
        self.max_steps = max_steps
        self.prefetches_to_device = device_prefetch is not None
        self.last_balance_stats: dict[str, Any] | None = None
        self._metadata_fn = metadata_fn
        self._bin_stats_fn = bin_stats_fn
        self._pack_fn = pack_fn
        self._collate_fn = collate_fn
        self._planner = planner
        self._transport = transport
        self._global_rank = global_rank
        self._data_rank = self.group_ranks.index(global_rank)
        self._iterator: _LocalBalancingIterator | None = None
        self._device_prefetch = device_prefetch
        self._device_batch = None
        self._balance_stats_callback = balance_stats_callback

    def __len__(self) -> int:
        """Return the configured step limit, or the source length when available."""
        return self.max_steps if self.max_steps is not None else len(self.local_dataloader)

    @property
    def step(self) -> int:
        """Return the number of delivered steps in the current iterator."""
        return self._iterator._step if self._iterator is not None else 0

    def __next__(self) -> Any:
        """Deliver a step without requiring a client-side iterator wrapper."""
        if self._iterator is None:
            self._iterator = _LocalBalancingIterator(self)
        return next(self._iterator)

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch selection to the local source before starting iteration."""
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        self._device_batch = None
        setter = getattr(self.local_dataloader, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Reject checkpointing until rank-local pending packs can be restored."""
        raise NotImplementedError("Local balancing does not yet support dataloader checkpoint/resume.")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Reject restoring incomplete locality/source state."""
        del state
        raise NotImplementedError("Local balancing does not yet support dataloader checkpoint/resume.")

    def __iter__(self) -> Iterator[Any]:
        """Yield constructed local microbatches after scoped metadata and sample exchange."""
        # Re-entering an unfinished iterator must not drop its speculative batch
        # after the stateful source has already advanced to the following step.
        if self._iterator is None or self._iterator.finished:
            if self._iterator is not None:
                self._iterator.close()
            self._iterator = _LocalBalancingIterator(self)
        return self._iterator

    def wait_for_prefetch(self) -> None:
        """Drain data communication before teardown, preserving the next batch.

        All ranks must call this at the same consumed-step boundary.
        """
        if self._iterator is not None:
            self._iterator.wait_for_prefetch()

    def take_device_microbatch(self, index: int) -> Any:
        """Take one staged microbatch just before training consumes it.

        Args:
            index: Microbatch index in the last delivered Host step.

        Returns:
            Device inputs with allocator lifetime recorded on the consumer stream.
        """
        if self._device_batch is None:
            raise RuntimeError("No device-prefetched step has been delivered.")
        return self._device_batch.take_microbatch(index)

    def close(self) -> None:
        """Drain the producer and release unconsumed Host/device views."""
        if self._iterator is not None:
            self._iterator.close()
        self._device_batch = None

    def _deliver_batch(self, result: _LocalBatch, step: int) -> Any:
        batch = result.data
        if self.prefetches_to_device:
            self._device_batch = batch
            # Logging and metering use the original Host view, without D2H.
            batch = batch.cpu_micro_batches
        if result.stats is not None and self._balance_stats_callback is not None and self._global_rank == 0:
            self._balance_stats_callback(result.stats, step, self.max_steps)
        return batch

    def _collate_and_stage(self, packed_bins: list[Any]) -> Any:
        batch = self._collate_fn(packed_bins)
        return self._device_prefetch(batch) if self._device_prefetch is not None else batch

    def _construct_batch(self, raw_bins: Sequence[Sequence[Any]], step: int) -> _LocalBatch:
        if not self.enable_balancing:
            # Disabled mode intentionally does not call metadata_fn or cost_model,
            # and creates no communication groups or sample payload codec buffers.
            return _LocalBatch(self._collate_and_stage([
                self._pack_fn(raw_bin, self.config.seq_len) for raw_bin in raw_bins
            ]))
        metadata, local_payloads = self._read_step(raw_bins, step)
        gathered = self._transport.all_gather_object(metadata, validate=False)
        plan, stats = self._plan_step(gathered, step)
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
            prepared = self._transport.prepare_exchange(outgoing, validate=False)
            received = self._transport.exchange_prepared(prepared, validate=False)
            received.update(retained_payloads)
        else:
            # Every rank sees the same plan and skips an unchanged exchange.
            received = local_payloads
        target = plan.local_batch_for(self._data_rank)
        batch = self._collate_and_stage([
            self._pack_fn([received[key] for key in packing_bin.sample_keys], self.config.seq_len)
            for packing_bin in target
        ])
        return _LocalBatch(batch, stats)

    def _read_step(self, raw_bins: Sequence[Sequence[Any]], step: int) -> tuple[tuple[Any, ...], dict[SampleKey, Any]]:
        key_bins = []
        entries = []
        payloads = {}
        for raw_bin in raw_bins:
            keys = []
            for sample in raw_bin:
                item_metadata = self._metadata_fn(sample)
                key = SampleKey(self._global_rank, step, len(entries))
                entries.append((key, item_metadata))
                payloads[key] = sample
                keys.append(key)
            key_bins.append(tuple(keys))
        return (tuple(key_bins), tuple(entries)), payloads

    @staticmethod
    def _selection(gathered: Sequence[Any]) -> StepSampleSelection:
        samples = []
        reference_bins = []
        for bins, entries in gathered:
            reference_bins.extend(bins)
            for key, metadata in entries:
                samples.append(BufferedSampleMetadata(key, metadata, len(samples)))
        return StepSampleSelection(tuple(samples), tuple(reference_bins), validate=False)

    def _plan_step(self, gathered: Sequence[Any], step: int) -> tuple[DistributedPackingPlan, dict[str, Any]]:
        result = None
        if self._global_rank == self._transport.planner_rank:
            selection = self._selection(gathered)
            plan = self._planner.plan(selection, step=step)
            stats = self._statistics(plan, step, gathered)
            if self._global_rank == 0:
                stats.update(self._bin_statistics(gathered, plan))
            result = (plan, stats)
        return self._transport.broadcast_from_planner(result, validate=False)

    def _bin_statistics(self, gathered: Sequence[Any], plan: DistributedPackingPlan) -> dict[str, Any]:
        """Summarize original and accepted bins on the planner, reusing gathered metadata."""
        # Gathered metadata predates cost-model evaluation. Reuse the accepted
        # plan's estimates for both layouts so before/after have the same units.
        metadata_by_key = {
            key: metadata
            for _, entries in gathered
            for key, metadata in entries
        }
        cost_by_key = {key: metadata.cost.llm for key, metadata in metadata_by_key.items()}
        before = []
        for key_bins, entries in gathered:
            metadata_by_key = dict(entries)
            before.append(tuple(
                {
                    **(self._bin_stats_fn(metadata_by_key[key] for key in keys) if self._bin_stats_fn else {}),
                    "samples": len(keys),
                    "seq_len": sum(metadata_by_key[key].pack_tokens for key in keys),
                    "cost": sum((cost_by_key[key] for key in keys), 0.0),
                }
                for keys in key_bins
            ))
        after = tuple(
            tuple(
                {
                    **(self._bin_stats_fn(metadata_by_key[key] for key in packing_bin.sample_keys)
                       if self._bin_stats_fn else {}),
                    "samples": len(packing_bin.sample_keys),
                    "seq_len": packing_bin.pack_tokens,
                    "cost": sum((cost_by_key[key] for key in packing_bin.sample_keys), 0.0),
                }
                for packing_bin in local_batch
            )
            for local_batch in plan.local_batches
        )
        return {"bins_before": tuple(before), "bins_after": after}

    def _statistics(
            self, plan: DistributedPackingPlan, step: int, gathered: Sequence[Any],
    ) -> dict[str, Any]:
        """Count cross-rank raw-sample transfers, excluding locally retained samples."""
        before = [0.0] * len(self.group_ranks)
        after = [0.0] * len(self.group_ranks)
        send_samples = [0] * len(self.group_ranks)
        recv_samples = [0] * len(self.group_ranks)
        moved = 0
        source_indices = {rank: index for index, rank in enumerate(self.group_ranks)}
        sample_costs = {
            key: metadata.cost.llm
            for _, entries in gathered
            for key, metadata in entries
        }
        for data_rank, local_batch in enumerate(plan.local_batches):
            for packing_bin in local_batch:
                for key in packing_bin.sample_keys:
                    source_index = source_indices[key.reader_rank]
                    # The plan carries placement, while source metadata carries
                    # the cost estimate used for diagnostics.
                    before[source_index] += sample_costs[key]
                    after[data_rank] += sample_costs[key]
                    if source_index != data_rank:
                        send_samples[source_index] += 1
                        recv_samples[data_rank] += 1
                        moved += 1
        mean = sum(before) / len(before)
        return {
            "step": step,
            "scope": self.balancing_scope,
            "objective": getattr(self._planner, "objective", "custom"),
            "group_ranks": self.group_ranks,
            "cost_before": tuple(before),
            "cost_after": tuple(after),
            "max_cost_before": max(before),
            "max_cost_after": max(after),
            "relative_spread_before": (max(before) - min(before)) / mean if mean else 0.0,
            "relative_spread_after": (max(after) - min(after)) / mean if mean else 0.0,
            "moved_samples": moved,
            "send_samples": tuple(send_samples),
            "recv_samples": tuple(recv_samples),
        }


def build_local_balancing_dataloader(
        local_dataloader: Iterable[Sequence[Sequence[Any]]],
        mesh: Any,
        config: DistributedDatasetConfig,
        *,
        metadata_fn: Callable[[Any], SampleMetadata],
        pack_fn: Callable[[Sequence[Any], int], Any],
        collate_fn: Callable[[Sequence[Any]], Any] = list,
        cost_model: CostModel | None = None,
        cost_model_id: str | None = None,
        balancing_scope: str = "node",
        balancing_objective: str = "balance",
        node_id: str | int | None = None,
        communication_device: Any = None,
        balancing_algorithm: Any = None,
        enable_balancing: bool = False,
        bin_stats_fn: Callable[[Iterable[SampleMetadata]], dict[str, Any]] | None = None,
        device_prefetch: DeviceStepPrefetcher | None = None,
        max_steps: int | None = None,
        balance_stats_callback: Callable[[dict[str, Any], int, int | None], None] | None = log_balance_stats,
) -> LocalBalancingDataLoader:
    """Wrap native online packing with local metadata planning and raw-sample A2A.

    Args:
        local_dataloader: Each yield is exactly ``config.local_batch_size``
            non-empty lists of transformed, uncollated raw samples. Native
            sampler, dataset transforms, worker settings, and buffering remain
            owned by this source. Its packs freeze current-step membership.
        mesh: Full named training mesh, with model parallel dimensions equal to 1.
        config: Sequence capacity, local pack count, hard stage budgets, DP
            dimension names and communication backends. Reader/worker/buffering
            settings are owned by ``local_dataloader``, not recreated here.
        metadata_fn: CPU raw-sample callback returning SampleMetadata.
        pack_fn: Existing task pack/collator called with received samples and seq_len.
        collate_fn: Assemble per-pack outputs into the trainer's local step format.
        cost_model: Optional deterministic workload estimate used by the planner.
        cost_model_id: Stable identity required when supplying a custom cost model.
        balancing_scope: ``node`` or actual named ``hsdp_shard`` mesh group.
        balancing_objective: ``balance`` minimizes DP load variance; ``makespan``
            prioritizes maximum predicted load, then variance. Both retain the
            original layout unless the selected objective improves.
        node_id: Optional override for node grouping; otherwise launcher/hostname.
        communication_device: Device used by the existing payload transport.
        balancing_algorithm: Optional planner instance exposing
            ``plan(StepSampleSelection, step=...)``; defaults to capacity-aware LPT.
            Custom planners own sample conservation, packing feasibility, and
            trainer/collective dimensions; this wrapper directly routes their output.
        enable_balancing: Opt-in switch. When false, preserve native packs and
            skip group creation, metadata extraction, planning and communication.
        bin_stats_fn: Optional CPU-only, read-only callback summarizing an iterable
            of sample metadata into a small serializable dictionary per bin. Runs
            only on rank zero, for original and accepted layouts; results are
            included in its node's existing statistics broadcast. The loader adds
            reserved ``samples``, ``seq_len`` and ``cost`` fields. Cost sums planner-estimated ``WorkloadCost.llm``
            for the bin, in the active cost model's units. Ignored when balancing
            is disabled. Does not change costs, placement, or communication count.
        device_prefetch: Optional final H2D stage, run after collation on the
            existing producer when double buffering is enabled. Iterator yields
            keep their Host view for metering; use ``take_device_microbatch``
            immediately before training consumes each staged microbatch.
        max_steps: Optional per-iterator step limit. No prefetch starts another
            step once this limit is reached.
        balance_stats_callback: Optional application log callback receiving
            ``(stats, one_based_step, max_steps)`` on global rank zero only,
            in the consuming thread, not DataLoader workers. Defaults to the
            built-in DP layout/movement/cost log; ``None`` disables logging.
            No extra statistics communication is added.

    Returns:
        Iterator preserving local microbatch count while balancing raw samples.

    Note:
        Double buffering starts the next full step automatically before returning
        the current batch (``ori`` timing); no model hooks or delayed trigger are needed.
        No checkpoint/resume is provided in this first version. Across different
        locality domains and within each domain, all ranks must consume the same
        number of steps. Source exhaustion and errors are not synchronized;
        a rank-local failure can leave peers waiting for the process-group timeout.
    """
    if not isinstance(enable_balancing, bool):
        raise ValueError("enable_balancing must be boolean.")
    if not enable_balancing:
        topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
        return LocalBalancingDataLoader(
            local_dataloader,
            config=config,
            metadata_fn=metadata_fn,
            pack_fn=pack_fn,
            collate_fn=collate_fn,
            planner=None,
            transport=None,
            global_rank=topology.global_rank,
            balancing_scope=balancing_scope,
            enable_balancing=False,
            bin_stats_fn=bin_stats_fn,
            device_prefetch=device_prefetch,
            max_steps=max_steps,
            balance_stats_callback=balance_stats_callback,
        )
    error = None
    identity = None
    try:
        effective_cost_id = resolve_cost_model_id(cost_model, cost_model_id)
        if balancing_objective not in ("balance", "makespan"):
            raise ValueError("balancing_objective must be 'balance' or 'makespan'.")
        if balancing_algorithm is not None and balancing_objective != "balance":
            raise ValueError("Configure the objective on the custom balancing_algorithm itself.")
        if not callable(metadata_fn) or not callable(pack_fn) or not callable(collate_fn):
            raise ValueError("metadata_fn, pack_fn and collate_fn must be callable.")
        if not hasattr(local_dataloader, "__iter__"):
            raise ValueError("local_dataloader must be iterable.")
        if balancing_algorithm is not None and not callable(getattr(balancing_algorithm, "plan", None)):
            raise ValueError("balancing_algorithm must expose plan(selection, step=...).")
        if config.dataset_reader_ranks is not None or config.planner_rank is not None:
            raise ValueError("Local balancing uses every scope rank as reader and its minimum rank as planner.")
        PackingConstraints(config.seq_len, config.oversized_policy, config.packing_budgets)
        identity = json.dumps({
            "config": asdict(config),
            "cost_model_id": effective_cost_id,
            "balancing_objective": balancing_objective,
            "max_steps": max_steps,
            "device_prefetch": device_prefetch is not None,
            "algorithm": None if balancing_algorithm is None else (
                type(balancing_algorithm).__module__ + "." + type(balancing_algorithm).__qualname__
            ),
        }, sort_keys=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    topology, groups = create_locality_groups(
        mesh,
        balancing_scope=balancing_scope,
        node_id=node_id,
        dp_dim_names=getattr(config, "dp_dim_names", None),
        cpu_backend=getattr(config, "cpu_backend", "gloo"),
        payload_backend=getattr(config, "payload_backend", None),
        communication_device=communication_device,
        build_identity=identity,
        local_error=error,
    )
    transport = DataPlaneTransport(groups, topology.global_rank, communication_device)
    planner = balancing_algorithm if balancing_algorithm is not None else LPTPackingPlanner(
        data_parallel_size=len(groups.data_plane_ranks),
        seq_len=config.seq_len,
        local_batch_size=config.local_batch_size,
        oversized_policy=config.oversized_policy,
        packing_budgets=config.packing_budgets,
        cost_model=cost_model,
        validate=False,
        objective=balancing_objective,
    )
    return LocalBalancingDataLoader(
        local_dataloader,
        config=config,
        metadata_fn=metadata_fn,
        pack_fn=pack_fn,
        collate_fn=collate_fn,
        planner=planner,
        transport=transport,
        global_rank=topology.global_rank,
        balancing_scope=balancing_scope,
        enable_balancing=True,
        bin_stats_fn=bin_stats_fn,
        device_prefetch=device_prefetch,
        max_steps=max_steps,
        balance_stats_callback=balance_stats_callback,
    )


__all__ = ["LocalBalancingDataLoader", "build_local_balancing_dataloader"]
