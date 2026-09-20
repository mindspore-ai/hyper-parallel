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
from hyper_parallel.distributed_data.balancing_algorithm import BalancingAlgorithm, resolve_balancing_algorithm
from hyper_parallel.distributed_data.cost_model import CostModel, resolve_cost_model
from hyper_parallel.distributed_data.device_prefetch import DeviceStepPrefetcher, _create_device_prefetcher
from hyper_parallel.distributed_data.locality import _create_locality_groups
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    PackingConstraints,
    SampleKey,
    SampleMetadata,
)
from hyper_parallel.distributed_data.transport import DataPlaneTransport, PayloadExchangeWork

if TYPE_CHECKING:
    from hyper_parallel.distributed_data.api import DistributedDatasetConfig


@dataclass(frozen=True)
class _LocalBatch:
    data: Any
    stats: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PreparedSourceStep:
    """CPU-only source data prepared before the foreground collective phase."""

    metadata: tuple[Any, ...]
    local_payloads: dict[SampleKey, Any]


@dataclass(frozen=True)
class _PendingBatch:
    """A planned step whose payload exchange may still be in flight."""

    plan: DistributedPackingPlan
    stats: dict[str, Any]
    retained_payloads: dict[SampleKey, Any]
    payload_work: PayloadExchangeWork | None


class _LocalBalancingIterator(Iterator[Any]):
    """Keep one step ahead with caller-owned accelerator collective launches."""

    def __init__(self, loader: LocalBalancingDataLoader) -> None:
        """Start source workers before entering the background pipeline."""
        self._source = iter(loader.local_dataloader)
        self._loader = loader
        self._step = 0
        self.finished = False
        self._thread: Thread | None = None
        self._result: _LocalBatch | None = None
        self._prepared_source_step: _PreparedSourceStep | None = None
        self._planned: Any = None
        self._phase = "source"
        self._error: BaseException | None = None

    def __next__(self) -> Any:
        """Consume a prepared step, completing omitted prefetch hooks inline."""
        if self.finished or self._limit_reached():
            self.finished = True
            raise StopIteration
        try:
            if self._loader._uses_synchronous_collectives:
                self.prefetch()
                if self.finished:
                    raise StopIteration
            elif self._thread is None:
                self._start_prefetch()
            self._join_worker()
            result = self._result
            self._result = None
            if result is None:
                raise RuntimeError("Local balancing prefetch completed without a batch.")
        except BaseException:
            self.finished = True
            raise
        self._step += 1
        self._loader.last_balance_stats = result.stats
        if not self._limit_reached():
            self._start_prefetch()
        return self._loader._deliver_batch(result, self._step)

    def _limit_reached(self) -> bool:
        return self._loader.max_steps is not None and self._step >= self._loader.max_steps

    def prefetch_plan(self) -> None:
        """Gather next-step metadata on the caller and start CPU planning.

        All ranks call at the same training boundary, after enqueueing compute
        and before host synchronization. No device-wide wait is introduced.
        """
        if not self._loader._uses_synchronous_collectives or self.finished or self._limit_reached():
            return
        if self._phase != "source":
            return
        if self._thread is None:
            self._start_prefetch()
        try:
            self._join_worker()
        except StopIteration:
            self.finished = True
            return
        with self._loader._data_stream_context():
            gathered = self._loader._transport.all_gather_object(self._prepared_source_step.metadata)
        self._phase = "plan"
        self._start_worker(self._prepare_plan, gathered)

    def prefetch(self) -> None:
        """Launch next-step payload exchange on the caller, then finish off-thread.

        All ranks must call at the same training boundary. Repeated calls
        before consumption are harmless; they do not advance source progress.
        """
        self.prefetch_plan()
        if not self._loader._uses_synchronous_collectives or self.finished or self._limit_reached():
            return
        if self._phase != "plan":
            return
        self._join_worker()
        with self._loader._data_stream_context():
            plan, stats = self._loader._transport.broadcast_from_planner(self._planned)
            pending = self._loader._begin_batch(self._prepared_source_step.local_payloads, plan, stats)
        self._prepared_source_step = None
        self._planned = None
        self._phase = "batch"
        self._start_worker(self._finish_batch, pending)

    def _prepare_plan(self, gathered: Sequence[Any]) -> None:
        self._planned = self._loader._make_plan(gathered, self._step)

    def _finish_batch(self, pending: _PendingBatch) -> None:
        with self._loader._data_stream_context():
            self._result = self._loader._finish_batch(pending)

    def _collect_batch(self, prepared: _PreparedSourceStep | None = None) -> _LocalBatch:
        if prepared is None:
            return self._loader._construct_batch(next(self._source), self._step)
        return self._loader._construct_batch((), self._step, prepared=prepared)

    def _start_prefetch(self) -> None:
        self._phase = "source"
        self._result = None
        self._prepared_source_step = None
        self._start_worker(self._run_prefetch)

    def _start_worker(self, callback: Callable[..., None], *args: Any) -> None:
        self._error = None
        self._thread = Thread(
            target=self._run_worker, args=(callback, args),
            name="hp-local-balance-prefetch", daemon=True,
        )
        self._thread.start()

    def _run_worker(self, callback: Callable[..., None], args: tuple[Any, ...]) -> None:
        try:
            callback(*args)
        except BaseException as exc:
            self._error = exc

    def _run_prefetch(self) -> None:
        if self._loader._uses_synchronous_collectives:
            raw_bins = next(self._source)
            metadata, payloads = self._loader._read_step(raw_bins, self._step)
            self._prepared_source_step = _PreparedSourceStep(metadata, payloads)
        else:
            self._result = self._collect_batch()

    def _join_worker(self) -> None:
        self.wait_for_prefetch()
        self._thread = None
        if self._error is not None:
            raise self._error

    def wait_for_prefetch(self) -> None:
        """Drain the active background phase without consuming its result."""
        if self._thread is not None:
            self._thread.join()

    def close(self) -> None:
        """Drain the producer before resetting the source or its epoch."""
        self.wait_for_prefetch()
        self.finished = True
        self._thread = None
        self._result = None
        self._prepared_source_step = None
        self._planned = None
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
            transport: DataPlaneTransport,
            global_rank: int,
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
        self.group_ranks = transport.ranks
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
        self._data_stream = None
        self._balance_stats_callback = balance_stats_callback

    @property
    def _uses_synchronous_collectives(self) -> bool:
        """Whether data collectives must stay on the training thread."""
        backend = getattr(self._transport, "communication_backend", "gloo").lower()
        return "hccl" in backend or "nccl" in backend

    def _data_stream_context(self) -> Any:
        device = getattr(self._transport, "communication_device", None)
        if device is None or torch.device(device).type == "cpu":
            return nullcontext()
        device = torch.device(device)
        accelerator = getattr(torch, device.type)
        accelerator.set_device(device)
        if self._data_stream is None:
            self._data_stream = accelerator.Stream(device=device)
        # Both launch and completion use this stream; neither waits on the
        # training stream just to move independent next-step data.
        return accelerator.stream(self._data_stream)

    def prefetch_plan(self) -> None:
        """Gather metadata and start planning at a rank-consistent compute boundary."""
        if self._iterator is None:
            self._iterator = _LocalBalancingIterator(self)
        self._iterator.prefetch_plan()

    def prefetch(self) -> None:
        """Launch next-step exchange before synchronizing current-step computation."""
        if self._iterator is None:
            self._iterator = _LocalBalancingIterator(self)
        self._iterator.prefetch()

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
        """Forward epoch selection to the local source before starting iteration.

        Args:
            epoch: Source epoch to start.
        """
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        if self._device_prefetch is not None:
            self._device_prefetch.close()
        self._device_batch = None
        setter = getattr(self.local_dataloader, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Reject checkpointing until rank-local pending packs can be restored."""
        raise NotImplementedError("Local balancing does not yet support dataloader checkpoint/resume.")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Reject restoring incomplete locality/source state.

        Args:
            state: Unsupported loader checkpoint.
        """
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
        if self._device_prefetch is not None:
            self._device_prefetch.close()
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

    def _construct_batch(
            self,
            raw_bins: Sequence[Sequence[Any]],
            step: int,
            *,
            prepared: _PreparedSourceStep | None = None,
    ) -> _LocalBatch:
        if prepared is None:
            metadata, local_payloads = self._read_step(raw_bins, step)
        else:
            metadata, local_payloads = prepared.metadata, prepared.local_payloads
        gathered = self._transport.all_gather_object(metadata)
        plan, stats = self._plan_step(gathered, step)
        return self._finish_batch(self._begin_batch(local_payloads, plan, stats))

    def _begin_batch(
            self, local_payloads: dict[SampleKey, Any], plan: DistributedPackingPlan, stats: dict[str, Any],
    ) -> _PendingBatch:
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

    def _finish_batch(self, pending: _PendingBatch) -> _LocalBatch:
        received = pending.retained_payloads
        if pending.payload_work is not None:
            received.update(pending.payload_work.wait())
        target = pending.plan.local_batch_for(self._data_rank)
        batch = self._collate_and_stage([
            self._pack_fn([received[key] for key in packing_bin.sample_keys], self.config.seq_len)
            for packing_bin in target
        ])
        return _LocalBatch(batch, pending.stats)

    def _read_step(self, raw_bins: Sequence[Sequence[Any]], step: int) -> tuple[tuple[Any, ...], dict[SampleKey, Any]]:
        if len(raw_bins) != self.config.local_batch_size or any(not raw_bin for raw_bin in raw_bins):
            raise ValueError("Each source step must contain local_batch_size non-empty raw-sample bins.")
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

    def _plan_step(self, gathered: Sequence[Any], step: int) -> tuple[DistributedPackingPlan, dict[str, Any]]:
        return self._transport.broadcast_from_planner(self._make_plan(gathered, step))

    def _make_plan(self, gathered: Sequence[Any], step: int) -> Any:
        result = None
        if self._global_rank == self._transport.planner_rank:
            samples = []
            reference_bins = []
            for bins, entries in gathered:
                reference_bins.extend(bins)
                for key, metadata in entries:
                    samples.append(BufferedSampleMetadata(key, metadata, len(samples)))
            plan = self._planner.plan(samples, reference_bins=reference_bins, step=step)
            stats = self._statistics(plan, step)
            if self._global_rank == 0:
                stats.update(self._bin_statistics(gathered, plan))
            result = (plan, stats)
        return result

    def _bin_statistics(self, gathered: Sequence[Any], plan: DistributedPackingPlan) -> dict[str, Any]:
        """Summarize original and accepted bins on the planner, reusing gathered metadata."""
        # Gathered metadata predates cost-model evaluation. Reuse the accepted
        # plan's estimates for both layouts so before/after have the same units.
        metadata_by_key = {
            key: metadata
            for _, entries in gathered
            for key, metadata in entries
        }
        cost_by_key = {key: cost.llm for key, cost in self._planner.last_sample_costs.items()}
        before = []
        for key_bins, _ in gathered:
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
            self, plan: DistributedPackingPlan, step: int,
    ) -> dict[str, Any]:
        """Count cross-rank raw-sample transfers, excluding locally retained samples."""
        before = [0.0] * len(self.group_ranks)
        after = [0.0] * len(self.group_ranks)
        send_samples = [0] * len(self.group_ranks)
        recv_samples = [0] * len(self.group_ranks)
        moved = 0
        source_indices = {rank: index for index, rank in enumerate(self.group_ranks)}
        sample_costs = {key: cost.llm for key, cost in self._planner.last_sample_costs.items()}
        for data_rank, local_batch in enumerate(plan.local_batches):
            for packing_bin in local_batch:
                for key in packing_bin.sample_keys:
                    source_index = source_indices[key.reader_rank]
                    before[source_index] += sample_costs[key]
                    after[data_rank] += sample_costs[key]
                    if source_index != data_rank:
                        send_samples[source_index] += 1
                        recv_samples[data_rank] += 1
                        moved += 1
        mean = sum(before) / len(before)
        return {
            "step": step,
            "scope": "node",
            "objective": self._planner.objective,
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
            **self._planner.last_balance_decision,
        }


def build_local_balancing_dataloader(
        local_dataloader: Iterable[Sequence[Sequence[Any]]],
        mesh: Any,
        config: DistributedDatasetConfig,
        *,
        metadata_fn: Callable[[Any], SampleMetadata],
        pack_fn: Callable[[Sequence[Any], int], Any],
        collate_fn: Callable[[Sequence[Any]], Any] = list,
        model_config: Any = None,
        cost_model: CostModel | None = None,
        balancing_algorithm: BalancingAlgorithm | None = None,
        device: Any = None,
        move_fn: Callable[[Any, Any], Any] | None = None,
        bin_stats_fn: Callable[[Iterable[SampleMetadata]], dict[str, Any]] | None = None,
        max_steps: int | None = None,
) -> LocalBalancingDataLoader:
    """Build the fixed node-local balancing pipeline over complete raw steps.

    Args:
        local_dataloader: Each yield contains exactly local_batch_size non-empty
            raw-sample bins. The source owns sampling and worker prefetch.
        mesh: WORLD-covering named mesh with model-parallel dimensions of size one.
        config: Packing limits and the minimum relative improvement required.
        metadata_fn: CPU-only per-sample metadata and cost features.
        pack_fn: Construct one packed sequence from its assigned raw samples.
        collate_fn: Assemble the local step as a sequence of microbatches.
        model_config: Effective backbone dimensions for the default FLOPs model.
            Required when cost_model is omitted.
        cost_model: Optional user estimate replacing the default cost model.
        balancing_algorithm: Optional sample assignment and objective policy.
            Costs and the improvement gate remain framework-owned.
        device: Training device; defaults to the current NPU or CUDA device.
            CPU-only execution keeps batches on the host.
        move_fn: Optional per-microbatch tensor mapping to the training device;
            use it to retain fields that must stay on CPU.
        bin_stats_fn: Optional CPU-only per-bin counters for the rank-zero log.
        max_steps: Stop before prefetching beyond the requested training steps.

    Returns:
        A local-step loader using the configured node-local communication
        backend, automatic one-step buffering and final H2D prefetch. Each step
        retains its source packs unless the candidate improves its objective by
        more than min_balance_gain.

    Note:
        All ranks must consume the same number of steps. Checkpoint/resume and
        nontrivial model parallelism are not supported by this local-step path.
        The iterator yields CPU views for metering; take_device_microbatch()
        hands the staged device view to training without another transfer.
    """
    error = None
    identity = None
    device_prefetch = None
    try:
        cost_model = resolve_cost_model(cost_model, model_config)
        balancing_algorithm = resolve_balancing_algorithm(balancing_algorithm)
        device_prefetch = _create_device_prefetcher(device, move_fn)
        if not all(callable(callback) for callback in (metadata_fn, pack_fn, collate_fn)):
            raise ValueError("metadata_fn, pack_fn and collate_fn must be callable.")
        if not hasattr(local_dataloader, "__iter__"):
            raise ValueError("local_dataloader must be iterable.")
        if config.dataset_reader_ranks is not None or config.planner_rank is not None:
            raise ValueError("Node-local balancing assigns readers and planners automatically.")
        PackingConstraints(config.seq_len, config.oversized_policy, config.packing_budgets)
        identity = json.dumps({
            "config": asdict(config),
            "policies": {
                name: {
                    "implementation": (
                        f"{policy.__module__}.{getattr(policy, '__qualname__', type(policy).__qualname__)}"
                    ),
                    "version": getattr(policy, field_name, None),
                }
                for name, policy, field_name in (
                    ("cost_model", cost_model, "model_id"),
                    ("balancing_algorithm", balancing_algorithm, "algorithm_id"),
                )
            },
            "max_steps": max_steps,
            "device_type": None if device_prefetch is None else device_prefetch.device.type,
        }, sort_keys=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    communication_device = device if config.communication_backend == "hccl" else None
    topology, groups = _create_locality_groups(
        mesh,
        dp_dim_names=config.dp_dim_names,
        build_identity=identity,
        local_error=error,
        communication_backend=config.communication_backend,
        communication_device=communication_device,
    )
    planner = DynamicPackingPlanner(
        data_parallel_size=len(groups.data_plane_ranks),
        seq_len=config.seq_len,
        local_batch_size=config.local_batch_size,
        oversized_policy=config.oversized_policy,
        packing_budgets=config.packing_budgets,
        cost_model=cost_model,
        balancing_algorithm=balancing_algorithm,
        min_balance_gain=config.min_balance_gain,
        validate=False,
    )
    return LocalBalancingDataLoader(
        local_dataloader,
        config=config,
        metadata_fn=metadata_fn,
        pack_fn=pack_fn,
        collate_fn=collate_fn,
        planner=planner,
        transport=DataPlaneTransport(
            groups,
            topology.global_rank,
            communication_device=communication_device,
        ),
        global_rank=topology.global_rank,
        bin_stats_fn=bin_stats_fn,
        device_prefetch=device_prefetch,
        max_steps=max_steps,
        balance_stats_callback=log_balance_stats,
    )


__all__ = ["LocalBalancingDataLoader", "build_local_balancing_dataloader"]
