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
"""Process-isolated balancing of externally selected raw-sample steps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from hyper_parallel.distributed_data.balance_logging import log_balance_stats
from hyper_parallel.distributed_data.balancing_algorithm import BalancingAlgorithm, resolve_balancing_algorithm
from hyper_parallel.distributed_data.cost_model import CostModel, resolve_cost_model
from hyper_parallel.distributed_data.device_prefetch import (
    DeviceStepPrefetcher, _create_device_prefetcher, _resolve_device,
)
from hyper_parallel.distributed_data.external_source import ExternalStepProducer, LocalBatch
from hyper_parallel.distributed_data.locality import _create_locality_groups
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.process_data import DataRequest, create_data_worker
from hyper_parallel.distributed_data.schema import PackingConstraints, SampleMetadata
from hyper_parallel.distributed_data.transport import DataGroups, DataPlaneTransport

if TYPE_CHECKING:
    from hyper_parallel.distributed_data.api import DistributedDatasetConfig


class LocalBalancingDataLoader(Iterator[Any]):
    """Balance external steps in the configured process and stage H2D on the caller."""

    def __init__(self, local_dataloader: Iterable[Sequence[Sequence[Any]]], *, config: DistributedDatasetConfig,
                 metadata_fn: Callable[[Any], SampleMetadata], pack_fn: Callable[[Sequence[Any], int], Any],
                 collate_fn: Callable[[Sequence[Any]], Any], planner: Any, transport: DataPlaneTransport,
                 global_rank: int, device_prefetch: DeviceStepPrefetcher | None = None,
                 groups: DataGroups | None = None,
                 balance_stats_callback: Callable[[dict[str, Any], int], None] | None = None) -> None:
        """Capture a serializable CPU source without starting workers or device work."""
        self.local_dataloader = local_dataloader
        self.config = config
        self.group_ranks = transport.ranks
        self.prefetches_to_device = device_prefetch is not None
        self.last_balance_stats = None
        self.last_host_batch = None
        self._source = ExternalStepProducer(local_dataloader, config=config, metadata_fn=metadata_fn,
                                            pack_fn=pack_fn, collate_fn=collate_fn, planner=planner,
                                            global_rank=global_rank, ranks=transport.ranks)
        self._transport = transport
        self._groups = groups
        self._global_rank = global_rank
        self._device_prefetch = device_prefetch
        self._balance_stats_callback = balance_stats_callback
        self._worker = None
        if not config.use_data_process:
            self._source.bind_transport(transport)
            self._worker = self._source
        self._host_request: DataRequest | None = None
        self._staged: LocalBatch | BaseException | None = None
        self._step = 0
        self._finished = False
        self._closed = False

    def _request_host(self) -> None:
        if self._host_request is None:
            if self._worker is None:
                self._worker = create_data_worker(self._source, self._transport)
            self._host_request = DataRequest(self._worker)

    def _stage_next(self, *, blocking: bool) -> None:
        """Receive one CPU result and launch its successor before submitting H2D."""
        if self._staged is not None:
            return
        if self.config.use_data_process:
            self._request_host()
        try:
            if self._host_request is not None:
                if not blocking and not self._host_request.is_ready():
                    return
                result = self._host_request.result()
                self._host_request = DataRequest(self._worker)
            else:
                result = next(self._source)
            if self._device_prefetch is not None:
                result = replace(result, data=self._device_prefetch(result.data))
            self._staged = result
        except BaseException as error:  # pylint: disable=broad-exception-caught
            # Deliver a callback or copy failure at its own step, after any preceding batch.
            self._staged = error

    def prefetch(self) -> None:
        """Start CPU lookahead and stage an available successor on the caller."""
        if self._closed or self._finished or not self.config.use_data_process:
            return
        self._request_host()
        if self.prefetches_to_device:
            self._stage_next(blocking=False)

    def prefetch_plan(self) -> None:
        """Retain the existing hook as an idempotent request for the complete pipeline."""
        self.prefetch()

    def __len__(self) -> int:
        """Return the external source length when available."""
        return len(self.local_dataloader)

    @property
    def step(self) -> int:
        """Return the number of delivered steps, excluding speculative progress."""
        return self._step

    def __next__(self) -> Any:
        """Consume one result and immediately request its successor."""
        if self._closed or self._finished:
            raise StopIteration
        self._stage_next(blocking=True)
        result, self._staged = self._staged, None
        if isinstance(result, BaseException):
            self._finished = True
            raise result
        self._step += 1
        self.last_balance_stats = result.stats
        self.last_host_batch = result.host_batch
        batch = result.data.take_batch() if self.prefetches_to_device else result.data
        if result.stats is not None and self._balance_stats_callback is not None and self._global_rank == 0:
            self._balance_stats_callback(result.stats, self._step)
        if self.prefetches_to_device and self.config.use_data_process:
            self._stage_next(blocking=self._step == 1)
        return batch

    def __iter__(self) -> LocalBalancingDataLoader:
        """Keep unfinished prefetch and re-enter an exhausted source in its owning process."""
        if self._finished and not self._closed:
            self._worker.reset_iteration()
            self._host_request = None
            self._staged = None
            self._finished = False
            self._step = 0
        return self

    def wait_for_prefetch(self) -> None:
        """Drain pending work without consuming or changing the delivered cursor."""
        if self._host_request is not None and self.prefetches_to_device and not self._finished:
            self._stage_next(blocking=True)
        if isinstance(self._staged, BaseException) and not isinstance(self._staged, StopIteration):
            raise self._staged
        if self._host_request is not None:
            try:
                self._host_request.result()
            except StopIteration:
                pass
        if self._device_prefetch is not None:
            self._device_prefetch.close()

    def set_epoch(self, epoch: int) -> None:
        """Reset external iteration after draining the previous source request.

        Args:
            epoch: Non-negative epoch to select before the next source read.
        """
        if self._closed:
            raise RuntimeError("Data loader is closed.")
        self.wait_for_prefetch()
        if self._worker is not None:
            self._worker.set_epoch(epoch)
        else:
            self._source.set_epoch(epoch)
        self._host_request = None
        self._staged = None
        self._finished = False
        self._step = 0
        self.last_host_batch = None
        self.last_balance_stats = None

    def state_dict(self) -> dict[str, Any]:
        """Preserve the external source's existing checkpoint ownership contract."""
        raise NotImplementedError("Local balancing does not yet support dataloader checkpoint/resume.")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Reject external checkpoint loading until pending source steps are specified.

        Args:
            state: External checkpoint payload; currently unsupported.
        """
        del state
        raise NotImplementedError("Local balancing does not yet support dataloader checkpoint/resume.")

    def close(self) -> None:
        """Drain device copies and release the data process."""
        with ExitStack() as cleanup:
            if self._groups is not None:
                cleanup.callback(self._groups.close)
            if self._closed:
                return
            self._closed = True
            if self._device_prefetch is not None:
                cleanup.callback(self._device_prefetch.close)
            if self._worker is not None:
                cleanup.callback(self._worker.close)
            # Delivery and explicit waits report speculative failures. Close must still release owned resources.
            with suppress(Exception):
                self.wait_for_prefetch()
            self._host_request = None
            self._staged = None
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
        device: Training device; defaults to CPU with Gloo, otherwise the current
            accelerator. Pass an explicit NPU/CUDA device for H2D with Gloo.

    Returns:
        A local-step loader using the configured node-local communication
        backend, automatic one-step buffering and final H2D prefetch. Windows
        that clear min_balance_gain use the candidate layout. The collator
        receives all packed bins of the source step together.

    Note:
        The external source controls iteration length and all ranks must consume
        the same number of steps. Checkpoint/resume and
        nontrivial model parallelism are not supported by this local-step path.
        Iteration returns device-ready batches. last_host_batch retains the CPU
        view for metering without a device-to-host copy.
    """
    if config.device_prefetch_depth != 1:
        raise ValueError("Configurable device_prefetch_depth requires a native batch_sampler.")
    if config.repack_sequences:
        raise ValueError("repack_sequences requires a native batch_sampler.")
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
    communication_device = device if config.communication_backend == "hccl" else None
    topology, groups = _create_locality_groups(
        mesh,
        dp_dim_names=config.dp_dim_names,
        balance_group_size=config.balance_group_size,
        communication_backend=config.communication_backend,
        communication_device=communication_device,
    )
    with groups.close_on_error():
        planner = DynamicPackingPlanner(
            data_parallel_size=len(groups.data_plane_ranks),
            seq_len=config.seq_len,
            # Window destinations are (micro-batch, rank), each holding one external bin.
            local_batch_size=config.local_batch_size if config.balance_window_size == 1 else 1,
            oversized_policy=config.oversized_policy,
            packing_budgets=config.packing_budgets,
            cost_model=cost_model,
            balancing_algorithm=balancing_algorithm,
            min_balance_gain=config.min_balance_gain,
            validate=False,
        )
        return LocalBalancingDataLoader(
            local_dataloader,
            groups=groups,
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
            balance_stats_callback=log_balance_stats if config.log_balance else None,
        )


__all__ = ["LocalBalancingDataLoader", "build_local_balancing_dataloader"]
