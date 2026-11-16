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

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING, Any

from hyper_parallel.distributed_data.balance_logging import log_balance_stats
from hyper_parallel.distributed_data.balancing_algorithm import BalancingAlgorithm, resolve_balancing_algorithm
from hyper_parallel.distributed_data.cost_model import CostModel, resolve_cost_model
from hyper_parallel.distributed_data.device_prefetch import (
    DeviceStepPrefetcher,
    _create_device_prefetcher,
    _resolve_device,
)
from hyper_parallel.distributed_data.locality import _create_locality_groups
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner, _policy_fingerprint
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    PackingConstraints,
    SampleKey,
    SampleMetadata,
)
from hyper_parallel.distributed_data.transport import DataPlaneTransport
from hyper_parallel.distributed_data.process_data import ProcessDataProducer, _cpu_rng_state, _restore_cpu_rng

if TYPE_CHECKING:
    from hyper_parallel.distributed_data.api import DistributedDatasetConfig


class _LocalStepProducer:
    """Own the source cursor, balancing policy and CPU batch construction."""

    def __init__(self, transport: Any, *, source: Any, config: Any, metadata_fn: Any,
                 pack_fn: Any, collate_fn: Any, planner: Any, global_rank: int) -> None:
        """Store only CPU source components; the transport is a parent proxy."""
        self.local_dataloader = source
        self.config = config
        self.group_ranks = transport.ranks
        self._transport = transport
        self._metadata_fn = metadata_fn
        self._pack_fn = pack_fn
        self._collate_fn = collate_fn
        self._planner = planner
        self._global_rank = global_rank
        self._data_rank = self.group_ranks.index(global_rank)
        self._source = None
        self._step = 0
        self._epoch = 0
        self._last_batch = None
        self._pending = None

    def __next__(self) -> Any:
        """Replay a pending batch or construct exactly one new source step."""
        if self._pending is not None:
            result, self._pending = self._pending, None
        else:
            if self._source is None:
                self._source = iter(self.local_dataloader)
            result = self._construct_batch()
        self._step += 1
        self._last_batch = result
        return result

    def set_epoch(self, epoch: int) -> None:
        """Reset the producer's iterator before forwarding the epoch."""
        self._source = None
        self._step = 0
        self._epoch = epoch
        self._pending = self._last_batch = None
        setter = getattr(self.local_dataloader, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def state_dict(self, consumed: int) -> dict[str, Any]:
        """Retain a produced but unconsumed batch beside the source cursor."""
        if self._step not in (consumed, consumed + 1):
            raise ValueError("Producer progress must be at most one step ahead of consumption.")
        return {
            "source": self.local_dataloader.state_dict(), "step": consumed, "epoch": self._epoch,
            "pending": self._last_batch if self._step == consumed + 1 else self._pending,
            "rng": _cpu_rng_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore source and CPU randomness before creating its iterator."""
        self._source = None
        self.local_dataloader.load_state_dict(state["source"])
        self._step, self._epoch = state["step"], state["epoch"]
        self._pending = state["pending"]
        self._last_batch = None
        _restore_cpu_rng(state["rng"])

    def close(self) -> None:
        """Release the source iterator and explicitly close an owned source."""
        self._source = None
        self._pending = self._last_batch = None
        closer = getattr(self.local_dataloader, "close", None)
        if callable(closer):
            closer()

    def _construct_batch(self) -> tuple[Any, dict[str, Any]]:
        metadata, payloads, error, exhausted = None, {}, None, False
        try:
            metadata, payloads = self._read_step(next(self._source), self._step)
        except StopIteration:
            exhausted = True
        except Exception as exc:  # Publish source failures in the existing metadata collective.
            error = f"{type(exc).__name__}: {exc}"
        statuses = self._transport.all_gather_object((exhausted, error, metadata))
        errors = [status[1] for status in statuses if status[1] is not None]
        if errors:
            raise RuntimeError(f"External data source failed: {errors}")
        if all(status[0] for status in statuses):
            raise StopIteration
        if any(status[0] for status in statuses):
            raise ValueError("External data sources must yield the same number of steps on every rank.")
        gathered = [status[2] for status in statuses]
        plan, stats = self._plan_step(gathered, self._step)
        if stats["moved_samples"]:
            payloads = self._route_payloads(plan, payloads)
        bins = plan.local_batch_for(self._data_rank)
        batch = self._collate_fn([
            self._pack_fn([payloads[key] for key in packing_bin.sample_keys], self.config.seq_len)
            for packing_bin in bins
        ])
        return batch, stats

    def _route_payloads(self, plan: DistributedPackingPlan, payloads: dict[SampleKey, Any]) -> dict[SampleKey, Any]:
        outgoing = {}
        retained = {}
        for data_rank, local_batch in enumerate(plan.local_batches):
            target = self.group_ranks[data_rank]
            owned = [(key, payloads[key]) for packing_bin in local_batch for key in packing_bin.sample_keys
                     if key.reader_rank == self._global_rank]
            if target == self._global_rank:
                retained.update(owned)
            else:
                outgoing[target] = owned
        retained.update(self._transport.exchange(outgoing))
        return retained

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
        result, error = None, None
        try:
            result = self._make_plan(gathered, step)
        except Exception as exc:  # Keep peers out of payload collectives when planning fails.
            error = f"{type(exc).__name__}: {exc}"
        result, error = self._transport.broadcast_from_planner((result, error))
        if error is not None:
            raise RuntimeError(f"External data planning failed: {error}")
        return result

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
                    "samples": len(keys),
                    "seq_len": sum(metadata_by_key[key].pack_tokens for key in keys),
                    "cost": sum((cost_by_key[key] for key in keys), 0.0),
                }
                for keys in key_bins
            ))
        after = tuple(
            tuple(
                {
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


class LocalBalancingDataLoader(Iterator[Any]):
    """Balance external steps in a CPU process with parent-owned collectives.

    The source and callbacks must be serializable before iteration starts and
    must produce CPU data. One speculative step overlaps training, including
    packing, data communication and H2D. All ranks consume matching steps.
    """

    def __init__(
            self, local_dataloader: Iterable[Sequence[Sequence[Any]]], *, config: DistributedDatasetConfig,
            metadata_fn: Callable[[Any], SampleMetadata], pack_fn: Callable[[Sequence[Any], int], Any],
            collate_fn: Callable[[Sequence[Any]], Any], planner: Any, transport: DataPlaneTransport,
            global_rank: int, device_prefetch: DeviceStepPrefetcher | None = None,
            max_steps: int | None = None,
            balance_stats_callback: Callable[[dict[str, Any], int, int | None], None] | None = None,
    ) -> None:
        """Store the source description without starting workers or device work."""
        if max_steps is not None and (not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None.")
        self.local_dataloader = local_dataloader
        self.config = config
        self.group_ranks = transport.ranks
        self.max_steps = max_steps
        self.prefetches_to_device = device_prefetch is not None
        self.last_balance_stats = None
        self.last_host_batch = None
        self._transport = transport
        self._global_rank = global_rank
        self._device_prefetch = device_prefetch
        self._balance_stats_callback = balance_stats_callback
        self._policy_identity = _policy_fingerprint(planner)
        self._factory = partial(
            _LocalStepProducer, source=local_dataloader, config=config, metadata_fn=metadata_fn,
            pack_fn=pack_fn, collate_fn=collate_fn, planner=planner, global_rank=global_rank,
        )
        self._producer = None
        self._step = 0
        self._epoch = 0
        self._finished = False
        self._resuming = False

    def _ensure_producer(self) -> ProcessDataProducer:
        if self._producer is None:
            self._producer = ProcessDataProducer(self._factory, self._transport, self._device_prefetch)
        return self._producer

    def __iter__(self) -> LocalBalancingDataLoader:
        """Keep repeated iteration at the same delivered cursor."""
        return self

    def __len__(self) -> int:
        """Return the configured step limit or the external source length."""
        return self.max_steps if self.max_steps is not None else len(self.local_dataloader)

    @property
    def step(self) -> int:
        """Return the number of delivered steps in this epoch."""
        return self._step

    def prefetch(self) -> None:
        """Prepare one step while current-step forward/backward is running."""
        if not self._finished and (self.max_steps is None or self._step < self.max_steps):
            self._ensure_producer().prefetch()

    def prefetch_plan(self) -> None:
        """Keep trainer hooks compatible with the single process pipeline."""
        self.prefetch()

    def __next__(self) -> Any:
        """Deliver a complete batch and immediately prefetch its successor."""
        if self._finished or (self.max_steps is not None and self._step >= self.max_steps):
            self._finished = True
            raise StopIteration
        try:
            batch, stats = self._ensure_producer().take()
        except StopIteration:
            self._finished = True
            raise
        self._step += 1
        self._resuming = False
        self.last_balance_stats = stats
        if self.prefetches_to_device:
            self.last_host_batch = batch.cpu_micro_batches
            ready = [batch.take_microbatch(index) for index in range(len(batch.cpu_micro_batches))]
            batch = tuple(ready) if isinstance(batch.cpu_micro_batches, tuple) else ready
        else:
            self.last_host_batch = batch
        if stats is not None and self._balance_stats_callback is not None and self._global_rank == 0:
            self._balance_stats_callback(stats, self._step, self.max_steps)
        self.prefetch()
        return batch

    def wait_for_prefetch(self) -> None:
        """Drain pending work without changing the consumed cursor."""
        if self._producer is not None:
            self._producer.wait()
        if self._device_prefetch is not None:
            self._device_prefetch.close()

    def set_epoch(self, epoch: int) -> None:
        """Reset the source in its owning process, preserving a restored same epoch."""
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if self._resuming and epoch == self._epoch:
            return
        self._ensure_producer().set_epoch(epoch)
        if self._device_prefetch is not None:
            self._device_prefetch.close()
        self._step, self._epoch = 0, epoch
        self._finished = self._resuming = False
        self.last_balance_stats = self.last_host_batch = None

    def _checkpoint_identity(self) -> tuple[Any, ...]:
        config = asdict(self.config)
        # Changing the transport does not change sample selection or packing.
        config.pop("communication_backend")
        return self.group_ranks, self._global_rank, config, self._policy_identity, self.max_steps

    def state_dict(self) -> dict[str, Any]:
        """Save the source's state and any prefetched batch at the consumed boundary."""
        if not callable(getattr(self.local_dataloader, "state_dict", None)):
            raise NotImplementedError("External checkpoint requires source.state_dict().")
        return {"identity": self._checkpoint_identity(), "finished": self._finished,
                "producer": self._ensure_producer().state_dict(self._step)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore matching source progress without losing the speculative batch."""
        if not callable(getattr(self.local_dataloader, "load_state_dict", None)):
            raise NotImplementedError("External checkpoint requires source.load_state_dict().")
        if state["identity"] != self._checkpoint_identity():
            raise ValueError("External checkpoint requires the same ranks and packing configuration.")
        self._ensure_producer().load_state_dict(state["producer"])
        self._step, self._epoch = state["producer"]["step"], state["producer"]["epoch"]
        self._finished = state["finished"]
        self._resuming = True
        self.last_balance_stats = self.last_host_batch = None

    def close(self) -> None:
        """Stop owned CPU workers after pending communication and H2D complete."""
        try:
            if self._producer is not None:
                self._producer.close()
        finally:
            if self._device_prefetch is not None:
                self._device_prefetch.close()
            self._finished = True
            self.last_host_batch = None


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
        max_steps: int | None = None,
) -> LocalBalancingDataLoader:
    """Build the fixed node-local balancing pipeline over complete raw steps.

    Args:
        local_dataloader: Each yield contains exactly local_batch_size non-empty
            raw-sample bins. The serializable, unstarted source owns sampling
            and CPU workers. Disable source pin_memory; HP pins final batches.
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
        device: Training device; defaults to CPU with Gloo, otherwise the current
            accelerator. Pass an explicit NPU/CUDA device for H2D with Gloo.
        max_steps: Stop before prefetching beyond the requested training steps.

    Returns:
        A local-step loader using the configured node-local communication
        backend, automatic one-step buffering and final H2D prefetch. Each step
        retains its source packs unless the candidate improves its objective by
        more than min_balance_gain.

    Note:
        All ranks must consume the same number of steps. This local-step path
        requires pure DP. Source state_dict/load_state_dict hooks enable
        checkpoint/resume, including one prefetched but unconsumed batch.
        Iteration returns device-ready batches. last_host_batch retains the CPU
        view for metering without a device-to-host copy.
    """
    cost_model = resolve_cost_model(cost_model, model_config)
    balancing_algorithm = resolve_balancing_algorithm(balancing_algorithm)
    device = _resolve_device(device, communication_backend=config.communication_backend)
    device_prefetch = _create_device_prefetcher(device)
    if not all(callable(callback) for callback in (metadata_fn, pack_fn, collate_fn)):
        raise ValueError("metadata_fn, pack_fn and collate_fn must be callable.")
    if not hasattr(local_dataloader, "__iter__"):
        raise ValueError("local_dataloader must be iterable.")
    if config.dataset_reader_ranks is not None or config.planner_rank is not None:
        raise ValueError("Node-local balancing assigns readers and planners automatically.")
    PackingConstraints(config.seq_len, config.oversized_policy, config.packing_budgets)
    communication_device = None
    if config.communication_backend == "hccl":
        communication_device = device_prefetch.device if device_prefetch is not None else device
    topology, groups = _create_locality_groups(
        mesh,
        dp_dim_names=config.dp_dim_names,
        balance_group_size=config.balance_group_size,
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
        device_prefetch=device_prefetch,
        max_steps=max_steps,
        balance_stats_callback=log_balance_stats,
    )


__all__ = ["LocalBalancingDataLoader", "build_local_balancing_dataloader"]
