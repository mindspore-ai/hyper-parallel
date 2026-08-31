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
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from multiprocessing.context import BaseContext
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.data_constructor import (
    PackingDataConstructor,
    default_collate_fn,
    default_pack_fn,
)
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner, OversizedPolicy
from hyper_parallel.distributed_data.schema import SampleMetadata
from hyper_parallel.distributed_data.sidecar import PlannedSampleLoader, SidecarMetadataReader
from hyper_parallel.distributed_data.dataset_reader import DatasetReader
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    create_data_groups,
    synchronize_build_preflight,
)

_CONFIG_DATALOADER_KWARGS = (
    "num_workers",
    "pin_memory",
    "prefetch_factor",
    "persistent_workers",
)
_OPTIONAL_DATALOADER_KWARGS = frozenset({
    "in_order",
    "multiprocessing_context",
    "pin_memory_device",
    "timeout",
    "worker_init_fn",
})
_SUPPORTED_DATALOADER_KWARGS = frozenset(_CONFIG_DATALOADER_KWARGS) | _OPTIONAL_DATALOADER_KWARGS
_MANAGED_DATALOADER_KWARGS = frozenset({
    "batch_sampler",
    "batch_size",
    "collate_fn",
    "dataset",
    "drop_last",
    "generator",
    "sampler",
    "shuffle",
})


@dataclass(frozen=True)
class DistributedDatasetConfig:
    """Configure sample planning, dynamic packing, and Dataset Reader workers.

    ``seq_len`` and ``local_batch_size`` are the only required sizing inputs.
    One iterator yield contains ``local_batch_size`` packed sequences on each
    DP rank. Optimizer gradient accumulation and global batch size remain
    Trainer concerns.

    Args:
        seq_len: Maximum token count in one non-oversized packed sequence.
        local_batch_size: Packed sequences constructed per DP rank and yield.
        dp_dim_names: Named mesh dimensions that define DP coordinates.
        dataset_reader_ranks: Optional Dataset Reader ranks. They read raw
            samples online or metadata only in sidecar mode. Defaults to the
            Data Constructor ranks.
        planner_rank: Optional centralized Planner rank. Defaults to the lowest
            Data Constructor rank.
        buffer_size_multiplier: Read-ahead token/sample target relative to one
            distributed batch. Larger values improve packing choices at higher
            Host-memory cost.
        max_buffered_samples: Per-reader candidate safety bound.
        oversized_policy: ``error`` by default; ``single`` explicitly permits
            one oversized sample to occupy a bin alone.
        drop_last: Whether to drop a tail with fewer than one sample per global
            packing bin. Only ``True`` is supported in this first version.
        shuffle: Whether Dataset Readers share one deterministic shuffled order.
        seed: Dataset Reader order and worker seed.
        num_workers: PyTorch workers per online Dataset Reader or sidecar
            constructor.
        pin_memory: Whether sample loader workers pin returned sample memory.
        prefetch_factor: Samples prefetched by each worker.
        persistent_workers: Whether workers persist for the loader lifetime.
        cpu_backend: torch.distributed backend for metadata/control and MP
            object delivery.
        payload_backend: Optional online payload A2A backend. With an
            accelerator communication device, the WORLD backend is used by default;
            otherwise this falls back to ``cpu_backend``.
    """

    seq_len: int
    local_batch_size: int
    dp_dim_names: tuple[str, ...] | None = None
    dataset_reader_ranks: tuple[int, ...] | None = None
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
    payload_backend: str | None = None

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
        self._validate_rank_tuple(self.dataset_reader_ranks, "dataset_reader_ranks")
        self._validate_name_tuple(self.dp_dim_names, "dp_dim_names")
        if self.planner_rank is not None and (
                not isinstance(self.planner_rank, int)
                or isinstance(self.planner_rank, bool)
                or self.planner_rank < 0
        ):
            raise ValueError("planner_rank must be a non-negative integer or None.")
        if not isinstance(self.cpu_backend, str) or not self.cpu_backend:
            raise ValueError("cpu_backend must be a non-empty string.")
        normalized_cpu_backend = self.cpu_backend.lower()
        if "hccl" in normalized_cpu_backend or "nccl" in normalized_cpu_backend:
            raise ValueError("cpu_backend must support CPU tensors and object collectives; use Gloo, not HCCL/NCCL.")
        if self.payload_backend is not None and (
                not isinstance(self.payload_backend, str) or not self.payload_backend
        ):
            raise ValueError("payload_backend must be a non-empty string or None.")

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
    dataset_reader_ranks = config.dataset_reader_ranks or topology.constructor_ranks
    unknown_ranks = set(dataset_reader_ranks) - set(topology.rank_list)
    if unknown_ranks:
        raise ValueError(f"dataset_reader_ranks are outside the root mesh: {sorted(unknown_ranks)}.")
    planner_rank = min(topology.constructor_ranks) if config.planner_rank is None else config.planner_rank
    data_plane_ranks = set(dataset_reader_ranks) | set(topology.constructor_ranks)
    if planner_rank not in data_plane_ranks:
        raise ValueError(f"planner_rank {planner_rank} must be a Dataset Reader or Data Constructor rank.")
    required_bins = topology.data_parallel_size * config.local_batch_size
    total_buffer_capacity = len(dataset_reader_ranks) * config.max_buffered_samples
    if total_buffer_capacity < required_bins:
        raise ValueError(
            f"Dataset Reader buffers can hold {total_buffer_capacity} samples, but one distributed yield requires at "
            f"least {required_bins}. Increase max_buffered_samples or Dataset Reader count."
        )
    return dataset_reader_ranks, planner_rank


def _normalize_dataloader_kwargs(
        config: DistributedDatasetConfig,
        dataloader_kwargs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[tuple[str, Any], ...]]:
    """Resolve validated PyTorch DataLoader execution options and fingerprint."""
    if dataloader_kwargs is None:
        supplied_options: dict[str, Any] = {}
    else:
        if not isinstance(dataloader_kwargs, Mapping):
            raise ValueError("dataloader_kwargs must be a mapping or None.")
        try:
            supplied_options = dict(dataloader_kwargs)
        except Exception as exc:
            raise ValueError(f"dataloader_kwargs cannot be copied: {exc}") from exc
    non_string_keys = [key for key in supplied_options if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError(f"dataloader_kwargs keys must be strings, but got {non_string_keys!r}.")
    managed_keys = sorted(set(supplied_options) & _MANAGED_DATALOADER_KWARGS)
    if managed_keys:
        raise ValueError(
            f"dataloader_kwargs cannot override distributed sampling, batching, collation, or deterministic "
            f"worker-seeding options "
            f"{managed_keys}."
        )
    unsupported_keys = sorted(set(supplied_options) - _SUPPORTED_DATALOADER_KWARGS)
    if unsupported_keys:
        raise ValueError(f"dataloader_kwargs contains unsupported options {unsupported_keys}.")

    options = {name: getattr(config, name) for name in _CONFIG_DATALOADER_KWARGS}
    options.update(supplied_options)
    num_workers = options["num_workers"]
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError(f"dataloader_kwargs num_workers must be a non-negative integer, but got {num_workers!r}.")
    for name in ("pin_memory", "persistent_workers"):
        if not isinstance(options[name], bool):
            raise ValueError(f"dataloader_kwargs {name} must be boolean, but got {options[name]!r}.")
    prefetch_factor = options["prefetch_factor"]
    if prefetch_factor is not None and (
            not isinstance(prefetch_factor, int)
            or isinstance(prefetch_factor, bool)
            or prefetch_factor < 1
    ):
        raise ValueError("dataloader_kwargs prefetch_factor must be a positive integer or None.")
    if num_workers == 0 and prefetch_factor is not None:
        raise ValueError("dataloader_kwargs prefetch_factor requires num_workers > 0.")
    if options["persistent_workers"] and num_workers == 0:
        raise ValueError("dataloader_kwargs persistent_workers=True requires num_workers > 0.")

    timeout = options.get("timeout", 0)
    try:
        normalized_timeout = float(timeout)
    except (TypeError, ValueError, OverflowError):
        normalized_timeout = math.nan
    if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(normalized_timeout)
            or normalized_timeout < 0
    ):
        raise ValueError(f"dataloader_kwargs timeout must be finite and non-negative, but got {timeout!r}.")
    if num_workers == 0 and timeout != 0:
        raise ValueError("dataloader_kwargs timeout must be zero when num_workers=0.")
    worker_init_fn = options.get("worker_init_fn")
    if worker_init_fn is not None and not callable(worker_init_fn):
        raise ValueError("dataloader_kwargs worker_init_fn must be callable or None.")
    multiprocessing_context = options.get("multiprocessing_context")
    if multiprocessing_context is not None:
        if num_workers == 0:
            raise ValueError("dataloader_kwargs multiprocessing_context requires num_workers > 0.")
        if not isinstance(multiprocessing_context, (str, BaseContext)):
            raise ValueError("dataloader_kwargs multiprocessing_context is invalid.")
        context_fingerprint = _multiprocessing_context_fingerprint(multiprocessing_context)
        if context_fingerprint not in torch.multiprocessing.get_all_start_methods():
            raise ValueError(
                f"dataloader_kwargs multiprocessing_context={context_fingerprint!r} is not supported."
            )
    pin_memory_device = options.get("pin_memory_device", "")
    if not isinstance(pin_memory_device, str):
        raise ValueError("dataloader_kwargs pin_memory_device must be a string.")
    in_order = options.get("in_order", True)
    if in_order is not True:
        raise ValueError("dataloader_kwargs in_order must remain True for deterministic sample routing.")

    fingerprint_values = {
        "in_order": in_order,
        "multiprocessing_context": _multiprocessing_context_fingerprint(multiprocessing_context),
        "num_workers": num_workers,
        "persistent_workers": options["persistent_workers"],
        "pin_memory": options["pin_memory"],
        "pin_memory_device": pin_memory_device,
        "prefetch_factor": prefetch_factor,
        "timeout": normalized_timeout,
        "worker_init_fn": worker_init_fn is not None,
    }
    return options, tuple(sorted(fingerprint_values.items()))


def _multiprocessing_context_fingerprint(context: Any) -> str | None:
    """Return a stable process-context identity without retaining the object."""
    if context is None or isinstance(context, str):
        return context
    try:
        start_method = context.get_start_method()
    except Exception as exc:
        raise ValueError(f"dataloader_kwargs multiprocessing_context cannot report its start method: {exc}") from exc
    return start_method


def _config_fingerprint(
        config: DistributedDatasetConfig,
        dataset_reader_ranks: tuple[int, ...],
        planner_rank: int,
        *,
        dataloader_fingerprint: tuple[tuple[str, Any], ...],
        sidecar_mode: bool,
        communication_device_type: str | None,
        uses_default_pack: bool,
        uses_default_collate: bool,
) -> str:
    stable_config = asdict(config)
    for name in _CONFIG_DATALOADER_KWARGS:
        stable_config.pop(name)
    stable_config["dataloader_options"] = dataloader_fingerprint
    stable_config["dataset_reader_ranks"] = dataset_reader_ranks
    stable_config["planner_rank"] = planner_rank
    stable_config["sidecar_mode"] = sidecar_mode
    stable_config["communication_device_type"] = communication_device_type
    stable_config["uses_default_pack"] = uses_default_pack
    stable_config["uses_default_collate"] = uses_default_collate
    return hashlib.sha256(repr(sorted(stable_config.items())).encode("utf-8")).hexdigest()[:24]


def _normalize_communication_device(communication_device: Any) -> tuple[torch.device | None, str | None]:
    """Normalize a rank-local A2A device without fingerprinting its local index."""
    if communication_device is None:
        return None, None
    try:
        device = torch.device(communication_device)
    except Exception as exc:
        raise ValueError(f"communication_device is invalid: {communication_device!r}.") from exc
    return device, device.type


def _build_fingerprint(topology: DataTopology, config_fingerprint: str) -> str:
    identity = (topology.fingerprint, config_fingerprint)
    return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:24]


def build_distributed_dataloader(
        dataset: Any | None,
        mesh: Any,
        config: DistributedDatasetConfig,
        *,
        metadata_fn: Callable[[Any], SampleMetadata] | None = None,
        metadata: Sequence[SampleMetadata] | None = None,
        dataloader_kwargs: Mapping[str, Any] | None = None,
        pack_fn: Callable[[Sequence[Any], int], Any] | None = None,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
        communication_device: Any = None,
) -> DistributedDataLoader:
    """Build a sample-balanced distributed DataLoader from a raw Dataset.

    Online mode uses ``metadata_fn`` after a Dataset Reader materializes each
    sample, then routes selected payloads to target Data Constructors. Sidecar
    mode uses ``metadata`` before any Dataset read; after planning, each target
    constructor directly reads only its assigned Dataset indices and therefore
    skips payload A2A. ``metadata`` and Dataset indices must be one-to-one.

    Args:
        dataset: Online mode requires the mapping-style Dataset on Dataset Reader
            ranks. Sidecar mode requires it on Data Constructor ranks. Other
            ranks may pass the same object or ``None``.
        mesh: Named root HyperParallel or PyTorch DeviceMesh.
        config: Dynamic packing, service-rank, and worker configuration.
        metadata_fn: Convert one materialized raw sample to SampleMetadata in
            online mode. Mutually exclusive with ``metadata``.
        metadata: Optional shared sidecar sequence on Dataset Reader ranks. Entry
            ``metadata[index]`` must describe ``dataset[index]``.
        dataloader_kwargs: Optional PyTorch DataLoader execution options. These
            override the worker options retained in ``config``. Sampling,
            batching, shuffling, and DataLoader collation remain internally
            managed and cannot be overridden.
        pack_fn: Optionally construct one model-specific packed sequence. The
            default preserves each planned bin as a raw-sample tuple.
        collate_fn: Optionally collate ``local_batch_size`` packed sequences.
            The default preserves the bins as a tuple.
        communication_device: Optional rank-local device used by online payload
            A2A. With NCCL/HCCL this is normally ``cuda:<local_rank>`` or
            ``npu:<local_rank>``. Omit it for CPU/Gloo payload exchange.

    Returns:
        Stateful collective iterator yielding constructed local batches.

    Note:
        Checkpoint replay requires deterministic mapping-Dataset access for a
        given index and epoch; arbitrary worker-side RNG state is not captured.
        Dataset Reader ranks must expose equal logical Dataset or metadata
        lengths. In sidecar mode,
        constructor Dataset lengths must also match the shared metadata length.
    """
    sidecar_mode = metadata_fn is None
    topology = None
    dataset_reader_ranks = None
    planner_rank = None
    dataset_reader = None
    sidecar_reader = None
    direct_sample_loader = None
    normalized_dataloader_kwargs = None
    dataloader_fingerprint = None
    additional_dataloader_kwargs = None
    planner = None
    constructor = None
    config_fingerprint = None
    normalized_communication_device = None
    communication_device_type = None
    reader_size = None
    direct_dataset_size = None
    local_error = None
    try:
        if not isinstance(config, DistributedDatasetConfig):
            raise ValueError(f"config must be DistributedDatasetConfig, but got {type(config)}.")
        if metadata_fn is not None and not callable(metadata_fn):
            raise ValueError("metadata_fn must be callable or None.")
        if metadata_fn is not None and metadata is not None:
            raise ValueError("Provide either online metadata_fn or sidecar metadata, but not both.")
        if pack_fn is not None and not callable(pack_fn):
            raise ValueError("pack_fn must be callable or None.")
        if collate_fn is not None and not callable(collate_fn):
            raise ValueError("collate_fn must be callable or None.")
        normalized_dataloader_kwargs, dataloader_fingerprint = _normalize_dataloader_kwargs(
            config,
            dataloader_kwargs,
        )
        additional_dataloader_kwargs = {
            name: value
            for name, value in normalized_dataloader_kwargs.items()
            if name not in _CONFIG_DATALOADER_KWARGS
        }
        normalized_communication_device, communication_device_type = _normalize_communication_device(
            communication_device
        )
        uses_default_pack = pack_fn is None or pack_fn is default_pack_fn
        uses_default_collate = collate_fn is None or collate_fn is default_collate_fn
        effective_pack_fn = default_pack_fn if uses_default_pack else pack_fn
        effective_collate_fn = default_collate_fn if uses_default_collate else collate_fn
        topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
        dataset_reader_ranks, planner_rank = _resolve_service_ranks(topology, config)
        if topology.global_rank in dataset_reader_ranks:
            reader_idx = dataset_reader_ranks.index(topology.global_rank)
            if sidecar_mode:
                if metadata is None:
                    raise ValueError(f"Sidecar Dataset Reader rank {topology.global_rank} must provide metadata.")
                sidecar_reader = SidecarMetadataReader(
                    metadata,
                    reader_rank=topology.global_rank,
                    reader_idx=reader_idx,
                    reader_count=len(dataset_reader_ranks),
                    seq_len=config.seq_len,
                    shuffle=config.shuffle,
                    seed=config.seed,
                )
                reader_size = len(metadata)
            else:
                if dataset is None:
                    raise ValueError(f"Online Dataset Reader rank {topology.global_rank} must provide a Dataset.")
                dataset_reader = DatasetReader(
                    dataset,
                    metadata_fn,
                    reader_rank=topology.global_rank,
                    reader_idx=reader_idx,
                    reader_count=len(dataset_reader_ranks),
                    seq_len=config.seq_len,
                    shuffle=config.shuffle,
                    seed=config.seed,
                    num_workers=normalized_dataloader_kwargs["num_workers"],
                    pin_memory=normalized_dataloader_kwargs["pin_memory"],
                    prefetch_factor=normalized_dataloader_kwargs["prefetch_factor"],
                    persistent_workers=normalized_dataloader_kwargs["persistent_workers"],
                    dataloader_kwargs=additional_dataloader_kwargs,
                )
                reader_size = len(dataset)
        if sidecar_mode and topology.is_constructor:
            if dataset is None:
                raise ValueError(f"Sidecar Data Constructor rank {topology.global_rank} must provide a Dataset.")
            direct_sample_loader = PlannedSampleLoader(
                dataset,
                num_workers=normalized_dataloader_kwargs["num_workers"],
                pin_memory=normalized_dataloader_kwargs["pin_memory"],
                prefetch_factor=normalized_dataloader_kwargs["prefetch_factor"],
                persistent_workers=normalized_dataloader_kwargs["persistent_workers"],
                seed=config.seed,
                dataloader_kwargs=additional_dataloader_kwargs,
            )
            direct_dataset_size = len(dataset)
            if reader_size is not None and reader_size != direct_dataset_size:
                raise ValueError(
                    f"Sidecar metadata length {reader_size} does not match Dataset length {direct_dataset_size}."
                )
        planner = DynamicPackingPlanner(
            data_parallel_size=topology.data_parallel_size,
            seq_len=config.seq_len,
            local_batch_size=config.local_batch_size,
            oversized_policy=config.oversized_policy,
        )
        constructor = PackingDataConstructor(effective_pack_fn, effective_collate_fn, seq_len=config.seq_len)
        config_fingerprint = _config_fingerprint(
            config,
            dataset_reader_ranks,
            planner_rank,
            dataloader_fingerprint=dataloader_fingerprint,
            sidecar_mode=sidecar_mode,
            communication_device_type=communication_device_type,
            uses_default_pack=uses_default_pack,
            uses_default_collate=uses_default_collate,
        )
    except Exception as exc:  # Every WORLD rank must fail before subgroup creation.
        local_error = f"{type(exc).__name__}: {exc}"

    synchronize_build_preflight(
        build_fingerprint=(
            _build_fingerprint(topology, config_fingerprint)
            if topology is not None and config_fingerprint is not None
            else None
        ),
        is_reader=(
            topology is not None
            and dataset_reader_ranks is not None
            and topology.global_rank in dataset_reader_ranks
        ),
        reader_size=reader_size,
        is_direct_reader=(
            sidecar_mode
            and topology is not None
            and topology.is_constructor
        ),
        direct_dataset_size=direct_dataset_size,
        sidecar_mode=sidecar_mode,
        local_error=local_error,
    )
    if (
            topology is None
            or dataset_reader_ranks is None
            or planner_rank is None
            or planner is None
            or constructor is None
            or config_fingerprint is None
            or normalized_dataloader_kwargs is None
            or dataloader_fingerprint is None
            or additional_dataloader_kwargs is None
    ):
        raise ValueError("Distributed DataLoader build preflight completed without validated components.")
    groups = create_data_groups(
        topology,
        dataset_reader_ranks,
        planner_rank,
        cpu_backend=config.cpu_backend,
        payload_backend=config.payload_backend,
        communication_device=normalized_communication_device,
        enable_payload_exchange=not sidecar_mode,
    )

    return DistributedDataLoader(
        topology=topology,
        dataset_reader_ranks=dataset_reader_ranks,
        dataset_reader=dataset_reader,
        sidecar_reader=sidecar_reader,
        direct_sample_loader=direct_sample_loader,
        sidecar_mode=sidecar_mode,
        planner=planner,
        data_constructor=constructor,
        data_plane=DataPlaneTransport(
            groups,
            topology.global_rank,
            communication_device=normalized_communication_device,
        ),
        model_transport=ModelParallelTransport(topology, groups),
        buffer_size_multiplier=config.buffer_size_multiplier,
        max_buffered_samples=config.max_buffered_samples,
        config_fingerprint=config_fingerprint,
    )


__all__ = ["DistributedDatasetConfig", "build_distributed_dataloader"]
