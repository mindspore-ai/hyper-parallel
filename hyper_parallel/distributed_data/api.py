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

from hyper_parallel.distributed_data.batch_sampler import (
    BatchSamplerReader,
    native_sampler_fingerprint,
    preserve_sample,
)
from hyper_parallel.distributed_data.data_constructor import (
    PackingDataConstructor,
    default_collate_fn,
    default_pack_fn,
)
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner, OversizedPolicy
from hyper_parallel.distributed_data.schema import PackingConstraints, SampleMetadata
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.dataset_reader import _validate_worker_options
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
            samples online or metadata only in metadata mode. Defaults to the
            Data Constructor ranks.
        dataset_already_sharded: Legacy external-reader configuration flag.
            Native BatchSampler requires False because it owns DP slicing;
            external readers manage their own partitioning.
        planner_rank: Optional centralized Planner rank. Defaults to the lowest
            Data Constructor rank.
        buffer_size_multiplier: Legacy compatibility option. Step boundaries
            are supplied by BatchSampler or external readers, not read-ahead.
        oversized_policy: ``error`` by default; ``single`` explicitly permits
            one oversized sample to occupy a bin alone.
        drop_last: Whether to drop a tail with fewer than one sample per global
            packing bin. Only ``True`` is supported in this first version.
        shuffle: Must be False with native BatchSampler, which owns shuffling.
            External readers also manage their own order.
        seed: Dataset Reader order and worker seed.
        num_workers: PyTorch workers per online Dataset Reader or plan-aware
            sample loader in metadata mode.
        pin_memory: Whether sample loader workers pin returned sample memory.
        prefetch_factor: Samples prefetched by each worker.
        persistent_workers: Whether workers persist for the loader lifetime.
        double_buffer: Whether to prepare the next distributed local batch in
            a background thread while the trainer consumes the current batch.
        cpu_backend: torch.distributed backend for metadata/control and MP
            batch broadcast.
        payload_backend: Optional payload A2A backend. With an
            accelerator communication device, the WORLD backend is used by default;
            otherwise this falls back to ``cpu_backend``.
        packing_budgets: Optional per-packed-sequence additive hard limits for
            ``build_local_balancing_dataloader``. Each configured stage must
            occur in every sample's packing_costs. These limits are independent
            of cost-model balancing scores.
    """

    seq_len: int
    local_batch_size: int
    dp_dim_names: tuple[str, ...] | None = None
    dataset_reader_ranks: tuple[int, ...] | None = None
    dataset_already_sharded: bool = False
    planner_rank: int | None = None
    buffer_size_multiplier: float = 2.0
    oversized_policy: OversizedPolicy = "error"
    drop_last: bool = True
    shuffle: bool = False
    seed: int = 1234
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    double_buffer: bool = False
    min_balance_gain: float = 0.0
    cpu_backend: str = "gloo"
    payload_backend: str | None = None
    packing_budgets: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate topology-independent configuration boundaries."""
        self._validate_integer_fields()
        self._validate_buffer_size_multiplier()
        self._validate_policy_and_flags()
        _validate_worker_options({name: getattr(self, name) for name in _CONFIG_DATALOADER_KWARGS})
        self._validate_rank_tuple(self.dataset_reader_ranks, "dataset_reader_ranks")
        self._validate_name_tuple(self.dp_dim_names, "dp_dim_names")
        self._validate_planner_rank()
        self._validate_backends()
        PackingConstraints(self.seq_len, self.oversized_policy, self.packing_budgets)

    def _validate_integer_fields(self) -> None:
        for name in ("seq_len", "local_batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, but got {self.seed!r}.")

    def _validate_buffer_size_multiplier(self) -> None:
        multiplier = self.buffer_size_multiplier
        invalid_multiplier = (
            not isinstance(multiplier, (int, float))
            or isinstance(multiplier, bool)
            or multiplier < 1.0
            or (isinstance(multiplier, float) and not math.isfinite(multiplier))
        )
        if invalid_multiplier:
            raise ValueError("buffer_size_multiplier must be a number greater than or equal to 1.0.")

    def _validate_policy_and_flags(self) -> None:
        if self.oversized_policy not in ("error", "single"):
            raise ValueError("oversized_policy must be 'error' or 'single'.")
        for name in (
                "dataset_already_sharded",
                "drop_last",
                "shuffle",
                "double_buffer",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        if (
            not isinstance(self.min_balance_gain, (int, float))
            or isinstance(self.min_balance_gain, bool)
            or not math.isfinite(self.min_balance_gain)
            or not 0.0 <= self.min_balance_gain < 1.0
        ):
            raise ValueError("min_balance_gain must be in [0, 1).")
        if not self.drop_last:
            raise ValueError(
                "Dynamic distributed packing currently requires drop_last=True so every DP rank receives the "
                "same number of non-empty packing bins."
            )

    def _validate_planner_rank(self) -> None:
        if self.planner_rank is not None and (
                not isinstance(self.planner_rank, int)
                or isinstance(self.planner_rank, bool)
                or self.planner_rank < 0
        ):
            raise ValueError("planner_rank must be a non-negative integer or None.")

    def _validate_backends(self) -> None:
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
    return dataset_reader_ranks, planner_rank


def _normalize_dataloader_kwargs(
        config: DistributedDatasetConfig,
        dataloader_kwargs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[tuple[str, Any], ...]]:
    """Resolve validated DataLoader execution options and fingerprint."""
    supplied_options = _copy_dataloader_kwargs(dataloader_kwargs)
    _validate_dataloader_option_keys(supplied_options)
    options = {name: getattr(config, name) for name in _CONFIG_DATALOADER_KWARGS}
    options.update(supplied_options)
    _validate_worker_options(options)
    normalized_timeout = _validate_optional_dataloader_options(options)
    fingerprint_values = {
        "in_order": options.get("in_order", True),
        "multiprocessing_context": _multiprocessing_context_fingerprint(options.get("multiprocessing_context")),
        "num_workers": options["num_workers"],
        "persistent_workers": options["persistent_workers"],
        "pin_memory": options["pin_memory"],
        "pin_memory_device": options.get("pin_memory_device", ""),
        "prefetch_factor": options["prefetch_factor"],
        "timeout": normalized_timeout,
        "worker_init_fn": options.get("worker_init_fn") is not None,
    }
    return options, tuple(sorted(fingerprint_values.items()))


def _copy_dataloader_kwargs(dataloader_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    if dataloader_kwargs is None:
        return {}
    if not isinstance(dataloader_kwargs, Mapping):
        raise ValueError("dataloader_kwargs must be a mapping or None.")
    try:
        return dict(dataloader_kwargs)
    except Exception as exc:
        raise ValueError(f"dataloader_kwargs cannot be copied: {exc}") from exc


def _validate_dataloader_option_keys(supplied_options: Mapping[str, Any]) -> None:
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


def _validate_optional_dataloader_options(options: Mapping[str, Any]) -> float:
    normalized_timeout = _validate_dataloader_timeout(options)
    worker_init_fn = options.get("worker_init_fn")
    if worker_init_fn is not None and not callable(worker_init_fn):
        raise ValueError("dataloader_kwargs worker_init_fn must be callable or None.")
    _validate_multiprocessing_context(options)
    pin_memory_device = options.get("pin_memory_device", "")
    if not isinstance(pin_memory_device, str):
        raise ValueError("dataloader_kwargs pin_memory_device must be a string.")
    if options.get("in_order", True) is not True:
        raise ValueError("dataloader_kwargs in_order must remain True for deterministic sample routing.")
    return normalized_timeout


def _validate_dataloader_timeout(options: Mapping[str, Any]) -> float:
    num_workers = options["num_workers"]
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
    return normalized_timeout


def _validate_multiprocessing_context(options: Mapping[str, Any]) -> None:
    multiprocessing_context = options.get("multiprocessing_context")
    if multiprocessing_context is None:
        return
    if options["num_workers"] == 0:
        raise ValueError("dataloader_kwargs multiprocessing_context requires num_workers > 0.")
    if not isinstance(multiprocessing_context, (str, BaseContext)):
        raise ValueError("dataloader_kwargs multiprocessing_context is invalid.")
    context_fingerprint = _multiprocessing_context_fingerprint(multiprocessing_context)
    if context_fingerprint not in torch.multiprocessing.get_all_start_methods():
        raise ValueError(f"dataloader_kwargs multiprocessing_context={context_fingerprint!r} is not supported.")


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
        metadata_mode: bool,
        communication_device_type: str | None,
        uses_default_pack: bool,
        uses_default_collate: bool,
) -> str:
    stable_config = asdict(config)
    if config.packing_budgets is None:
        stable_config.pop("packing_budgets")
    else:
        stable_config["packing_budgets"] = tuple(sorted(config.packing_budgets.items()))
    for name in _CONFIG_DATALOADER_KWARGS:
        stable_config.pop(name)
    stable_config["dataloader_options"] = dataloader_fingerprint
    stable_config["dataset_reader_ranks"] = dataset_reader_ranks
    stable_config["planner_rank"] = planner_rank
    # Keep the serialized key stable so existing checkpoint fingerprints still match.
    stable_config["sidecar_mode"] = metadata_mode
    stable_config["communication_device_type"] = communication_device_type
    stable_config["uses_default_pack"] = uses_default_pack
    stable_config["uses_default_collate"] = uses_default_collate
    return hashlib.sha256(repr(sorted(stable_config.items())).encode("utf-8")).hexdigest()[:24]


def _normalize_communication_device(communication_device: Any) -> torch.device | None:
    """Normalize a rank-local A2A device without fingerprinting its local index."""
    if communication_device is None:
        return None
    try:
        return torch.device(communication_device)
    except Exception as exc:
        raise ValueError(f"communication_device is invalid: {communication_device!r}.") from exc


def _build_fingerprint(topology: DataTopology, config_fingerprint: str) -> str:
    identity = (topology.fingerprint, config_fingerprint)
    return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:24]


def _resolve_metadata_mode(
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        external_step_reader: Any | None,
) -> bool:
    """Resolve one metadata-mode flag when external Readers are rank-local."""
    local_flags = (external_step_reader is not None, metadata is not None)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        external_present = local_flags[0]
        metadata_present = local_flags[1]
    else:
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_flags)
        external_present = any(item[0] for item in gathered)
        metadata_present = any(item[1] for item in gathered)
    if external_present:
        return False
    return metadata_fn is None or metadata_present


@dataclass
class _BuildState:
    """Rank-local components and partial validation results needed across build stages."""

    metadata_mode: bool
    topology: DataTopology | None = None
    dataset_reader_ranks: tuple[int, ...] | None = None
    planner_rank: int | None = None
    dataset_reader: Any | BatchSamplerReader | None = None
    metadata_reader: BatchSamplerReader | None = None
    direct_sample_loader: PlannedSampleLoader | None = None
    planner: DynamicPackingPlanner | None = None
    constructor: PackingDataConstructor | None = None
    config_fingerprint: str | None = None
    communication_device: torch.device | None = None
    reader_size: int | None = None
    direct_dataset_size: int | None = None
    local_error: str | None = None
    external_step_mode: bool = False

    @property
    def is_reader(self) -> bool:
        """Return whether this rank owns a Dataset Reader."""
        return (
            self.topology is not None
            and self.dataset_reader_ranks is not None
            and self.topology.global_rank in self.dataset_reader_ranks
        )


class _DatasetMetadataView(Sequence[SampleMetadata]):
    """Expose Dataset-provided metadata through a sequence contract."""

    def __init__(self, dataset: Any) -> None:
        """Store a source Dataset with a metadata-only lookup method."""
        self._dataset = dataset

    def __len__(self) -> int:
        """Return the aligned source Dataset length."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> SampleMetadata:
        """Return metadata without materializing the source payload."""
        return self._dataset.get_sample_metadata(index)


def _infer_dataset_metadata(
        dataset: Any | None,
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
) -> Sequence[SampleMetadata] | None:
    """Use metadata from an indexed Dataset when the caller omits metadata callbacks."""
    if metadata_fn is not None or metadata is not None or dataset is None:
        return metadata
    get_sample_metadata = getattr(dataset, "get_sample_metadata", None)
    supports_metadata = bool(getattr(dataset, "requires_distributed_packing", False))
    if supports_metadata and callable(get_sample_metadata):
        return _DatasetMetadataView(dataset)
    return None


def _validate_builder_callbacks(
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        pack_fn: Callable[[Sequence[Any], int], Any] | None,
        collate_fn: Callable[[Sequence[Any]], Any] | None,
) -> None:
    if metadata_fn is not None and not callable(metadata_fn):
        raise ValueError("metadata_fn must be callable or None.")
    if metadata_fn is not None and metadata is not None:
        raise ValueError("Provide either metadata or metadata_fn, but not both.")
    if pack_fn is not None and not callable(pack_fn):
        raise ValueError("pack_fn must be callable or None.")
    if collate_fn is not None and not callable(collate_fn):
        raise ValueError("collate_fn must be callable or None.")


def _configure_external_step_reader(
        state: _BuildState,
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        external_step_reader: Any | None = None,
) -> None:
    if external_step_reader is not None:
        if metadata is not None or metadata_fn is not None:
            raise ValueError("external_step_reader cannot be combined with metadata or metadata_fn.")
        if not state.is_reader:
            raise ValueError("external_step_reader may only be provided on Dataset Reader ranks.")
        required_methods = (
            "prepare_next_step",
            "metadata",
            "selected_payloads",
            "commit",
            "state_dict",
            "load_state_dict",
            "set_epoch",
        )
        missing = [name for name in required_methods if not callable(getattr(external_step_reader, name, None))]
        for name in ("exhausted", "reference_bins"):
            if not hasattr(external_step_reader, name):
                missing.append(name)
        if missing:
            raise ValueError(f"external_step_reader is missing methods: {missing}")
        state.dataset_reader = external_step_reader
        state.external_step_mode = True
        return
    if state.metadata_mode:
        raise ValueError(
            "Metadata mode requires batch_sampler to define step/sample boundaries; "
            "metadata-only streaming selection has been removed. "
            "For online loading, provide external_step_reader."
        )
    if state.is_reader:
        raise ValueError(
            "Online mode requires external_step_reader; provide a Reader that emits one complete local step."
        )


def _configure_batch_sampler_sources(
        state: _BuildState,
        dataset: Any,
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        config: DistributedDatasetConfig,
        batch_sampler: Any,
        loader_options: dict[str, Any],
) -> str:
    """Use native DP sampler owners as Readers without applying a second stride."""
    if state.dataset_reader_ranks != state.topology.constructor_ranks:
        raise ValueError("Native batch_sampler currently requires Dataset Readers on the Data Constructor ranks.")
    if config.shuffle or config.dataset_already_sharded:
        raise ValueError(
            "Native batch_sampler owns DP slicing and shuffle; disable shuffle and dataset_already_sharded."
        )
    sampler_fingerprint = native_sampler_fingerprint(
        batch_sampler, data_rank=state.topology.data_rank,
        dp_size=state.topology.data_parallel_size, local_batch_size=config.local_batch_size,
    )
    if not state.is_reader:
        return sampler_fingerprint
    if dataset is None:
        raise ValueError("Native batch_sampler requires a mapping Dataset on each Data Constructor.")
    sample_loader = PlannedSampleLoader(dataset, seed=config.seed, **loader_options)
    sample_loader.set_epoch(batch_sampler.epoch)
    state.reader_size = len(dataset)
    if state.metadata_mode:
        if not callable(getattr(metadata, "__getitem__", None)) or len(metadata) != len(dataset):
            raise ValueError("Native batch_sampler metadata must align with the mapping Dataset.")
        state.direct_sample_loader = sample_loader
        state.direct_dataset_size = len(dataset)
    elif metadata_fn is None:
        raise ValueError("Native batch_sampler online mode requires metadata_fn.")
    reader = BatchSamplerReader(
        batch_sampler, reader_rank=state.topology.global_rank,
        policy_fingerprint=sampler_fingerprint,
        metadata=metadata, metadata_fn=metadata_fn,
        sample_loader=None if state.metadata_mode else sample_loader,
    )
    if state.metadata_mode:
        state.metadata_reader = reader
    else:
        state.dataset_reader = reader
    return sampler_fingerprint


def _populate_build_state(
        state: _BuildState,
        dataset: Any | None,
        mesh: Any,
        config: DistributedDatasetConfig,
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        dataloader_kwargs: Mapping[str, Any] | None,
        pack_fn: Callable[[Sequence[Any], int], Any] | None,
        collate_fn: Callable[[Sequence[Any]], Any] | None,
        communication_device: Any,
        batch_sampler: Any = None,
        external_step_reader: Any | None = None,
) -> None:
    if not isinstance(config, DistributedDatasetConfig):
        raise ValueError(f"config must be DistributedDatasetConfig, but got {type(config)}.")
    if external_step_reader is not None and batch_sampler is not None:
        raise ValueError("external_step_reader and batch_sampler are mutually exclusive.")
    if external_step_reader is None:
        metadata = _infer_dataset_metadata(dataset, metadata_fn, metadata)
    elif metadata is not None or metadata_fn is not None:
        raise ValueError("external_step_reader cannot be combined with metadata or metadata_fn.")
    _validate_builder_callbacks(metadata_fn, metadata, pack_fn, collate_fn)
    normalized_options, dataloader_fingerprint = _normalize_dataloader_kwargs(
        config,
        dataloader_kwargs,
    )
    loader_options = {name: normalized_options[name] for name in _CONFIG_DATALOADER_KWARGS}
    loader_options["dataloader_kwargs"] = {
        name: value
        for name, value in normalized_options.items()
        if name not in _CONFIG_DATALOADER_KWARGS
    }
    state.communication_device = _normalize_communication_device(communication_device)
    uses_default_pack = pack_fn is None or pack_fn is default_pack_fn
    uses_default_collate = collate_fn is None or collate_fn is default_collate_fn
    effective_pack_fn = default_pack_fn if uses_default_pack else pack_fn
    if batch_sampler is not None:
        if pack_fn is not None:
            raise ValueError("Native batch_sampler preserves Dataset outputs; pack_fn must be omitted.")
        effective_pack_fn = preserve_sample
    effective_collate_fn = default_collate_fn if uses_default_collate else collate_fn
    state.topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
    state.dataset_reader_ranks, state.planner_rank = _resolve_service_ranks(state.topology, config)
    batch_sampler_fingerprint = None
    if batch_sampler is None:
        _configure_external_step_reader(
            state, metadata_fn, metadata,
            external_step_reader=external_step_reader,
        )
    else:
        batch_sampler_fingerprint = _configure_batch_sampler_sources(
            state, dataset, metadata_fn, metadata, config, batch_sampler, loader_options,
        )
    state.planner = DynamicPackingPlanner(
        data_parallel_size=state.topology.data_parallel_size,
        seq_len=config.seq_len,
        local_batch_size=config.local_batch_size,
        oversized_policy=config.oversized_policy,
        min_balance_gain=config.min_balance_gain,
    )
    state.constructor = PackingDataConstructor(effective_pack_fn, effective_collate_fn, seq_len=config.seq_len)
    state.config_fingerprint = _config_fingerprint(
        config,
        state.dataset_reader_ranks,
        state.planner_rank,
        dataloader_fingerprint=dataloader_fingerprint,
        metadata_mode=state.metadata_mode,
        communication_device_type=None if state.communication_device is None else state.communication_device.type,
        uses_default_pack=uses_default_pack,
        uses_default_collate=uses_default_collate,
    )
    if batch_sampler_fingerprint is not None:
        state.config_fingerprint += ":batch_sampler:" + batch_sampler_fingerprint


def _synchronize_build_state(state: _BuildState, config: DistributedDatasetConfig) -> None:
    build_fingerprint = None
    if state.topology is not None and state.config_fingerprint is not None:
        build_fingerprint = _build_fingerprint(state.topology, state.config_fingerprint)
    # Invalid configs must still participate in WORLD build preflight synchronization.
    dataset_already_sharded = isinstance(config, DistributedDatasetConfig) and config.dataset_already_sharded
    is_direct_reader = state.metadata_mode and state.topology is not None and state.topology.is_constructor
    state.external_step_mode = synchronize_build_preflight(
        build_fingerprint=build_fingerprint,
        is_reader=state.is_reader,
        reader_size=state.reader_size,
        is_direct_reader=is_direct_reader,
        direct_dataset_size=state.direct_dataset_size,
        metadata_mode=state.metadata_mode,
        dataset_already_sharded=dataset_already_sharded,
        local_error=state.local_error,
        external_step_mode=state.external_step_mode,
    )
    # Consumer-only ranks have no reader object, but must share selection mode
    # and checkpoint identity with the ranks that produce their model batch.
    if state.external_step_mode and state.config_fingerprint is not None:
        state.config_fingerprint += ":external_step"


def _require_build_state(state: _BuildState) -> None:
    required_components = (
        state.topology,
        state.dataset_reader_ranks,
        state.planner_rank,
        state.planner,
        state.constructor,
        state.config_fingerprint,
    )
    if any(component is None for component in required_components):
        raise ValueError("Distributed DataLoader build preflight completed without validated components.")


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
        batch_sampler: Any = None,
        external_step_reader: Any | None = None,
) -> DistributedDataLoader:
    """Build a sample-balanced distributed DataLoader.

    Online mode without ``batch_sampler`` requires ``external_step_reader``. The external Reader must
    emit one complete local step, including metadata and canonical pack
    boundaries; this loader then freezes the union of those samples, balances
    them across Data Constructors, and routes payloads when necessary.
    Metadata mode requires ``batch_sampler`` to define step/sample boundaries.
    It looks up ``metadata[index]`` before any Dataset read; target constructors
    then directly read their assigned indices from the shared Dataset and skip
    payload A2A. Metadata-only streaming selection is not supported.

    With ``batch_sampler``, native HP sampling owns step selection:
    one sampler yield per DP Constructor fixes one forward/backward round.
    Complete Dataset outputs are balanced without repacking their contents.

    Args:
        dataset: BatchSampler mode requires a shared mapping Dataset
            on Data Constructor ranks. In external-step mode, the Reader owns data
            loading and ``dataset`` may be ``None``. Other ranks may pass the
            same object or ``None``.
        mesh: Named root HyperParallel or native DeviceMesh.
        config: Dynamic packing, service-rank, and worker configuration.
        metadata_fn: Only used by ``batch_sampler`` mode to derive metadata
            from each Dataset output. It cannot be used by external-step mode.
        metadata: Precomputed metadata for BatchSampler mode on Dataset Reader ranks. Indexed
            source Datasets that implement ``get_sample_metadata`` provide this
            automatically when both metadata arguments are omitted. It is a
            shared global sequence. Entry ``metadata[index]`` must
            describe the corresponding ``dataset[index]``.
        dataloader_kwargs: Optional DataLoader execution options. These
            override the worker options retained in ``config``. Sampling,
            batching, shuffling, and DataLoader collation remain internally
            managed and cannot be overridden.
        pack_fn: Optionally construct one model-specific packed sequence. The
            default preserves each planned bin as a raw-sample tuple.
        collate_fn: Optionally collate ``local_batch_size`` packed sequences.
            The default preserves the bins as a tuple.
        communication_device: Optional rank-local device used by payload A2A.
            With NCCL/HCCL this is normally ``cuda:<local_rank>`` or
            ``npu:<local_rank>``. Omit it for CPU/Gloo payload exchange.
        batch_sampler: Optional native HP BatchSampler. Supply the rank-local
            sampler on every rank; only each DP Constructor advances it. Its
            next yield fixes local sample membership, with no second stride,
            shuffle, or dynamic selection. Each Dataset output stays whole in
            one bin and goes unchanged to ``collate_fn``. Readers must coincide
            with Constructors, ``drop_last`` must be true, and ``pack_fn`` must
            be omitted. Metadata entries must describe these Dataset indices,
            not underlying document indices. Checkpoint through this loader,
            not through the sampler's speculative prefetch cursor.
        external_step_reader: Required for online mode without ``batch_sampler``. It is a
            rank-local external producer that must
            expose ``prepare_next_step``, ``metadata``, ``reference_bins``,
            ``selected_payloads``, ``commit``, ``exhausted``, and checkpoint/epoch methods.
            One call to ``prepare_next_step`` supplies exactly one already-selected local
            step. HP preserves that step's union and only rebalances its target
            ranks.

    Returns:
        Stateful collective iterator yielding constructed local batches.

    Note:
        Checkpoint replay requires a deterministic online stream for a given
        epoch; arbitrary worker-side RNG state is not captured. Metadata and
        Dataset lengths must agree across all Data Constructor ranks. Metadata
        entries must describe deterministic, rank-independent Dataset outputs.
    """
    return _build_distributed_dataloader_impl(
        dataset,
        mesh,
        config,
        metadata_fn=metadata_fn,
        metadata=metadata,
        dataloader_kwargs=dataloader_kwargs,
        pack_fn=pack_fn,
        collate_fn=collate_fn,
        communication_device=communication_device,
        batch_sampler=batch_sampler,
        external_step_reader=external_step_reader,
    )


def _build_distributed_dataloader_impl(
        dataset: Any | None,
        mesh: Any,
        config: DistributedDatasetConfig,
        *,
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        dataloader_kwargs: Mapping[str, Any] | None,
        pack_fn: Callable[[Sequence[Any], int], Any] | None,
        collate_fn: Callable[[Sequence[Any]], Any] | None,
        communication_device: Any,
        batch_sampler: Any = None,
        external_step_reader: Any | None = None,
) -> DistributedDataLoader:
    metadata_mode = _resolve_metadata_mode(metadata_fn, metadata, external_step_reader)
    state = _BuildState(metadata_mode=metadata_mode)
    try:
        _populate_build_state(
            state,
            dataset,
            mesh,
            config,
            metadata_fn,
            metadata,
            dataloader_kwargs,
            pack_fn,
            collate_fn,
            communication_device,
            batch_sampler,
            external_step_reader,
        )
    except Exception as exc:  # Every WORLD rank must fail before subgroup creation.
        state.local_error = f"{type(exc).__name__}: {exc}"

    _synchronize_build_state(state, config)
    _require_build_state(state)
    groups = create_data_groups(
        state.topology,
        state.dataset_reader_ranks,
        state.planner_rank,
        cpu_backend=config.cpu_backend,
        payload_backend=config.payload_backend,
        communication_device=state.communication_device,
        enable_payload_exchange=not state.metadata_mode,
    )

    return DistributedDataLoader(
        batch_sampler_mode=batch_sampler is not None,
        external_step_mode=state.external_step_mode,
        initial_epoch=0 if batch_sampler is None else batch_sampler.epoch,
        topology=state.topology,
        dataset_reader_ranks=state.dataset_reader_ranks,
        dataset_reader=state.dataset_reader,
        metadata_reader=state.metadata_reader,
        direct_sample_loader=state.direct_sample_loader,
        metadata_mode=state.metadata_mode,
        planner=state.planner,
        data_constructor=state.constructor,
        data_plane=DataPlaneTransport(
            groups,
            state.topology.global_rank,
            communication_device=state.communication_device,
        ),
        model_transport=ModelParallelTransport(state.topology, groups),
        double_buffer=config.double_buffer,
        config_fingerprint=state.config_fingerprint,
    )


__all__ = ["DistributedDatasetConfig", "build_distributed_dataloader"]
