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
"""Public builder for sample-balanced distributed dynamic packing."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from hyper_parallel.distributed_data.data_constructor import PackingDataConstructor
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner, OversizedPolicy
from hyper_parallel.distributed_data.schema import SampleMetadata
from hyper_parallel.distributed_data.source_loader import SourceLoader
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    create_data_groups,
    synchronize_build_preflight,
)


@dataclass(frozen=True)
class DistributedDatasetConfig:
    """Configure sample planning, dynamic packing, and Source Loader workers.

    ``seq_len`` and ``local_batch_size`` are the only required sizing inputs.
    One iterator yield contains ``local_batch_size`` packed sequences on each
    DP rank. Optimizer gradient accumulation and global batch size remain
    Trainer concerns.

    Args:
        seq_len: Maximum token count in one non-oversized packed sequence.
        local_batch_size: Packed sequences constructed per DP rank and yield.
        dp_dim_names: Named mesh dimensions that define DP coordinates.
        source_loader_ranks: Optional raw-sample reader ranks. Defaults to one
            Data Constructor rank per DP coordinate.
        planner_rank: Optional centralized Planner rank. Defaults to the lowest
            Data Constructor rank.
        buffer_size_multiplier: Read-ahead token/sample target relative to one
            distributed batch. Larger values improve packing choices at higher
            Host-memory cost.
        max_buffered_samples: Per-Source payload safety bound.
        oversized_policy: ``error`` by default; ``single`` explicitly permits
            one oversized sample to occupy a bin alone.
        drop_last: Whether to drop a tail with fewer than one sample per global
            packing bin. Only ``True`` is supported in this first version.
        shuffle: Whether Source Loaders share one deterministic shuffled order.
        seed: Source order and worker seed.
        num_workers: PyTorch workers per Source Loader rank.
        pin_memory: Whether Source Loader workers pin returned sample memory.
        prefetch_factor: Samples prefetched by each worker.
        persistent_workers: Whether workers persist for the loader lifetime.
        cpu_backend: torch.distributed backend for metadata, payload, and MP
            delivery. ``gloo`` keeps the data plane in CPU memory.
    """

    seq_len: int
    local_batch_size: int
    dp_dim_names: tuple[str, ...] | None = None
    source_loader_ranks: tuple[int, ...] | None = None
    planner_rank: int | None = None
    buffer_size_multiplier: float = 2.0
    max_buffered_samples: int = 10_000
    oversized_policy: OversizedPolicy = "error"
    drop_last: bool = True
    shuffle: bool = False
    seed: int = 1234
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    cpu_backend: str = "gloo"

    def __post_init__(self) -> None:
        """Validate topology-independent configuration boundaries."""
        for name in ("seq_len", "local_batch_size", "max_buffered_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        for name in ("seed", "num_workers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, but got {value!r}.")
        multiplier = self.buffer_size_multiplier
        invalid_multiplier = (
            not isinstance(multiplier, (int, float))
            or isinstance(multiplier, bool)
            or multiplier < 1.0
            or (isinstance(multiplier, float) and not math.isfinite(multiplier))
        )
        if invalid_multiplier:
            raise ValueError("buffer_size_multiplier must be a number greater than or equal to 1.0.")
        if self.oversized_policy not in ("error", "single"):
            raise ValueError("oversized_policy must be 'error' or 'single'.")
        for name in ("drop_last", "shuffle", "pin_memory", "persistent_workers"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        if not self.drop_last:
            raise ValueError(
                "Dynamic distributed packing currently requires drop_last=True so every DP rank receives the "
                "same number of non-empty packing bins."
            )
        if self.prefetch_factor is not None and (
                not isinstance(self.prefetch_factor, int)
                or isinstance(self.prefetch_factor, bool)
                or self.prefetch_factor < 1
        ):
            raise ValueError("prefetch_factor must be a positive integer or None.")
        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0.")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0.")
        self._validate_rank_tuple(self.source_loader_ranks, "source_loader_ranks")
        self._validate_name_tuple(self.dp_dim_names, "dp_dim_names")
        if self.planner_rank is not None and (
                not isinstance(self.planner_rank, int)
                or isinstance(self.planner_rank, bool)
                or self.planner_rank < 0
        ):
            raise ValueError("planner_rank must be a non-negative integer or None.")
        if not isinstance(self.cpu_backend, str) or not self.cpu_backend:
            raise ValueError("cpu_backend must be a non-empty string.")

    @staticmethod
    def _validate_rank_tuple(value: tuple[int, ...] | None, name: str) -> None:
        if value is None:
            return
        if not isinstance(value, tuple) or not value:
            raise ValueError(f"{name} must be a non-empty tuple or None.")
        if any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in value):
            raise ValueError(f"{name} must contain non-negative integer ranks.")
        if len(value) != len(set(value)):
            raise ValueError(f"{name} must not contain duplicate ranks.")

    @staticmethod
    def _validate_name_tuple(value: tuple[str, ...] | None, name: str) -> None:
        if value is None:
            return
        if not isinstance(value, tuple) or not value or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"{name} must be a non-empty tuple of strings or None.")
        if len(value) != len(set(value)):
            raise ValueError(f"{name} must not contain duplicate names.")


def _resolve_service_ranks(
        topology: DataTopology,
        config: DistributedDatasetConfig,
) -> tuple[tuple[int, ...], int]:
    source_loader_ranks = config.source_loader_ranks or topology.constructor_ranks
    unknown_ranks = set(source_loader_ranks) - set(topology.rank_list)
    if unknown_ranks:
        raise ValueError(f"source_loader_ranks are outside the root mesh: {sorted(unknown_ranks)}.")
    planner_rank = min(topology.constructor_ranks) if config.planner_rank is None else config.planner_rank
    data_plane_ranks = set(source_loader_ranks) | set(topology.constructor_ranks)
    if planner_rank not in data_plane_ranks:
        raise ValueError(f"planner_rank {planner_rank} must be a Source Loader or Data Constructor rank.")
    required_bins = topology.data_parallel_size * config.local_batch_size
    total_buffer_capacity = len(source_loader_ranks) * config.max_buffered_samples
    if total_buffer_capacity < required_bins:
        raise ValueError(
            f"Source buffers can hold {total_buffer_capacity} samples, but one distributed yield requires at "
            f"least {required_bins}. Increase max_buffered_samples or Source Loader count."
        )
    return source_loader_ranks, planner_rank


def _config_fingerprint(
        config: DistributedDatasetConfig,
        source_loader_ranks: tuple[int, ...],
        planner_rank: int,
) -> str:
    stable_config = asdict(config)
    stable_config["source_loader_ranks"] = source_loader_ranks
    stable_config["planner_rank"] = planner_rank
    return hashlib.sha256(repr(sorted(stable_config.items())).encode("utf-8")).hexdigest()[:24]


def _build_fingerprint(topology: DataTopology, config_fingerprint: str) -> str:
    identity = (topology.fingerprint, config_fingerprint)
    return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:24]


def build_distributed_dataloader(
        dataset: Any | None,
        mesh: Any,
        config: DistributedDatasetConfig,
        *,
        metadata_fn: Callable[[Any], SampleMetadata],
        pack_fn: Callable[[Sequence[Any], int], Any],
        collate_fn: Callable[[Sequence[Any]], Any],
) -> DistributedDataLoader:
    """Build a sample-balanced distributed DataLoader from a raw Dataset.

    Source Loader ranks materialize individual ``Dataset.__getitem__`` results
    without batching or collation. ``metadata_fn`` runs after materialization;
    the Planner uses only that lightweight metadata. Selected raw payloads then
    move to target Data Constructors, where ``pack_fn`` runs once per planned
    sequence and ``collate_fn`` runs once per rank-local batch.

    Args:
        dataset: Mapping-style Dataset on Source Loader ranks. Other ranks may
            pass the same object or ``None``. Multiple datasets can be composed
            with ``ConcatDataset``, a Megatron BlendableDataset, or another
            mapping-style mixer before calling this builder.
        mesh: Named root HyperParallel or PyTorch DeviceMesh.
        config: Dynamic packing, service-rank, and worker configuration.
        metadata_fn: Convert one materialized raw sample to SampleMetadata.
        pack_fn: Pack an ordered raw-sample list under ``seq_len``.
        collate_fn: Collate ``local_batch_size`` packed sequences.

    Returns:
        Stateful collective iterator yielding user-collated local batches.

    Note:
        This is reactive online planning: Source Loaders read payloads before
        metadata is available. Ahead-of-fetch sidecar planning is a separate
        optimization and is not part of this API. Checkpoint replay of future
        samples requires deterministic mapping-Dataset access for a given
        index and epoch; arbitrary worker-side RNG state is not captured. All
        Source Loader ranks must receive replicas of the same logical Dataset;
        the builder collectively validates equal lengths, while index semantics
        remain part of the user Dataset contract.
    """
    topology = None
    source_loader_ranks = None
    planner_rank = None
    source_loader = None
    planner = None
    constructor = None
    config_fingerprint = None
    dataset_size = None
    local_error = None
    try:
        if not isinstance(config, DistributedDatasetConfig):
            raise ValueError(f"config must be DistributedDatasetConfig, but got {type(config)}.")
        if not callable(metadata_fn) or not callable(pack_fn) or not callable(collate_fn):
            raise ValueError("metadata_fn, pack_fn, and collate_fn must be callable.")
        topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
        source_loader_ranks, planner_rank = _resolve_service_ranks(topology, config)
        if topology.global_rank in source_loader_ranks:
            if dataset is None:
                raise ValueError(f"Source Loader rank {topology.global_rank} must provide a Dataset.")
            source_loader = SourceLoader(
                dataset,
                metadata_fn,
                source_rank=topology.global_rank,
                source_lane=source_loader_ranks.index(topology.global_rank),
                source_lane_count=len(source_loader_ranks),
                seq_len=config.seq_len,
                shuffle=config.shuffle,
                seed=config.seed,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers,
            )
            dataset_size = len(dataset)
        planner = DynamicPackingPlanner(
            data_parallel_size=topology.data_parallel_size,
            seq_len=config.seq_len,
            local_batch_size=config.local_batch_size,
            oversized_policy=config.oversized_policy,
        )
        constructor = PackingDataConstructor(pack_fn, collate_fn, seq_len=config.seq_len)
        config_fingerprint = _config_fingerprint(config, source_loader_ranks, planner_rank)
    except Exception as exc:  # Every WORLD rank must fail before subgroup creation.
        local_error = f"{type(exc).__name__}: {exc}"

    synchronize_build_preflight(
        build_fingerprint=(
            _build_fingerprint(topology, config_fingerprint)
            if topology is not None and config_fingerprint is not None
            else None
        ),
        is_source=(
            topology is not None
            and source_loader_ranks is not None
            and topology.global_rank in source_loader_ranks
        ),
        dataset_size=dataset_size,
        local_error=local_error,
    )
    if (
            topology is None
            or source_loader_ranks is None
            or planner_rank is None
            or planner is None
            or constructor is None
            or config_fingerprint is None
    ):
        raise ValueError("Distributed DataLoader build preflight completed without validated components.")
    groups = create_data_groups(
        topology,
        source_loader_ranks,
        planner_rank,
        cpu_backend=config.cpu_backend,
    )

    return DistributedDataLoader(
        topology=topology,
        source_loader_ranks=source_loader_ranks,
        source_loader=source_loader,
        planner=planner,
        data_constructor=constructor,
        data_plane=DataPlaneTransport(groups, topology.global_rank),
        model_transport=ModelParallelTransport(topology, groups),
        buffer_size_multiplier=config.buffer_size_multiplier,
        max_buffered_samples=config.max_buffered_samples,
        config_fingerprint=config_fingerprint,
    )


__all__ = ["DistributedDatasetConfig", "build_distributed_dataloader"]
