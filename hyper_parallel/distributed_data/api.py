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
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from multiprocessing.context import BaseContext
from typing import Any, Literal

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
from hyper_parallel.distributed_data.dataset import DistributedDataset
from hyper_parallel.distributed_data.dataset_dataloader import DatasetDataLoader
from hyper_parallel.distributed_data.device_prefetch import _create_device_prefetcher, _resolve_device
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.balancing_algorithm import BalancingAlgorithm
from hyper_parallel.distributed_data.cost_model import CostModel
from hyper_parallel.distributed_data.packed_balancing import LocalBalancingDataLoader, build_local_balancing_dataloader
from hyper_parallel.distributed_data.external_step import ExternalStepAdapter, ExternalStepSource
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner, OversizedPolicy
from hyper_parallel.distributed_data.schema import PackingConstraints, SampleMetadata
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.dataset_reader import _validate_worker_options
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    all_gather_control_object,
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
        min_balance_gain: Minimum relative improvement in the algorithm's
            objective. A candidate is accepted only when its gain is strictly
            greater; zero rejects equal or worse placements.
        packing_budgets: Optional per-packed-sequence additive hard limits.
            Each configured stage must
            occur in every sample's packing_costs. These limits are independent
            of cost-model balancing scores.
        communication_backend: Backend for HP data-plane collectives. ``hccl``
            is the default for NPU training and serializes control objects into
            accelerator tensors. ``gloo`` keeps control and CPU payload
            communication on Gloo. HCCL requires an NPU communication device.
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
    min_balance_gain: float = 0.0
    packing_budgets: dict[str, float] | None = None
    communication_backend: Literal["gloo", "hccl"] = "hccl"

    def __post_init__(self) -> None:
        """Validate topology-independent configuration boundaries."""
        self._validate_integer_fields()
        self._validate_buffer_size_multiplier()
        self._validate_policy_and_flags()
        _validate_worker_options({name: getattr(self, name) for name in _CONFIG_DATALOADER_KWARGS})
        self._validate_rank_tuple(self.dataset_reader_ranks, "dataset_reader_ranks")
        self._validate_name_tuple(self.dp_dim_names, "dp_dim_names")
        self._validate_planner_rank()
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
        if self.communication_backend not in ("gloo", "hccl"):
            raise ValueError("communication_backend must be 'gloo' or 'hccl'.")
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
        external_step_source: ExternalStepSource | None,
        communication_backend: str = "hccl",
        communication_device: Any = None,
) -> bool:
    """Resolve one metadata-mode flag when external Readers are rank-local."""
    local_flags = (external_step_reader is not None or external_step_source is not None, metadata is not None)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        external_present = local_flags[0]
        metadata_present = local_flags[1]
    else:
        startup_group = None
        if communication_backend == "gloo":
            startup_group = torch.distributed.new_group(
                ranks=list(range(torch.distributed.get_world_size())), backend="gloo"
            )
        gathered = all_gather_control_object(
            local_flags,
            group=startup_group,
            device=communication_device,
            backend=communication_backend,
        )
        external_present = any(item[0] for item in gathered)
        metadata_present = any(item[1] for item in gathered)
    if external_present:
        return False
    return metadata_fn is None or metadata_present


@dataclass
class _BuildState:
    """Rank-local components and partial validation results needed across build stages."""

    metadata_mode: bool
    model_config: Any = None
    cost_model: CostModel | None = None
    balancing_algorithm: BalancingAlgorithm | None = None
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
        if not hasattr(external_step_reader, "exhausted"):
            missing.append("exhausted")
        if not any(hasattr(external_step_reader, name) for name in ("original_metadatas", "reference_bins")):
            missing.append("original_metadatas")
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


def _configure_external_step_source(
        state: _BuildState,
        source: ExternalStepSource | None,
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        pack_fn: Callable[[Sequence[Any], int], Any],
        collate_fn: Callable[[Sequence[Any]], Any],
        config: DistributedDatasetConfig,
) -> None:
    """Wrap a source-only producer with HP's generic Reader lifecycle."""
    if source is None:
        if state.is_reader:
            raise ValueError("Dataset Reader ranks must provide external_step_source.")
        return
    if not state.is_reader:
        raise ValueError("external_step_source may only be provided on Dataset Reader ranks.")
    if metadata_fn is None:
        raise ValueError("external_step_source requires metadata_fn.")
    state.dataset_reader = ExternalStepAdapter(
        source,
        reader_rank=state.topology.global_rank,
        local_batch_size=config.local_batch_size,
        seq_len=config.seq_len,
        metadata_fn=metadata_fn,
        pack_fn=pack_fn,
        collate_fn=collate_fn,
    )
    state.external_step_mode = True


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


def _resolve_constructor_callbacks(
        batch_sampler: Any,
        pack_fn: Callable[[Sequence[Any], int], Any] | None,
        collate_fn: Callable[[Sequence[Any]], Any] | None,
) -> tuple[Callable[[Sequence[Any], int], Any], Callable[[Sequence[Any]], Any], bool, bool]:
    """Resolve constructor callbacks and their stable fingerprint flags."""
    uses_default_pack = pack_fn is None or pack_fn is default_pack_fn
    uses_default_collate = collate_fn is None or collate_fn is default_collate_fn
    effective_pack_fn = default_pack_fn if uses_default_pack else pack_fn
    if batch_sampler is not None:
        if pack_fn is not None:
            raise ValueError("Native batch_sampler preserves Dataset outputs; pack_fn must be omitted.")
        effective_pack_fn = preserve_sample
    effective_collate_fn = default_collate_fn if uses_default_collate else collate_fn
    return effective_pack_fn, effective_collate_fn, uses_default_pack, uses_default_collate


def _configure_step_sources(
        state: _BuildState,
        dataset: Any | None,
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        config: DistributedDatasetConfig,
        loader_options: dict[str, Any],
        effective_pack_fn: Callable[[Sequence[Any], int], Any],
        effective_collate_fn: Callable[[Sequence[Any]], Any],
        batch_sampler: Any,
        external_step_reader: Any | None,
        external_step_source: ExternalStepSource | None,
) -> str | None:
    """Configure one mutually exclusive online or native step source."""
    if batch_sampler is None:
        if external_step_source is not None:
            _configure_external_step_source(
                state,
                external_step_source,
                metadata_fn,
                effective_pack_fn,
                effective_collate_fn,
                config,
            )
        else:
            _configure_external_step_reader(
                state, metadata_fn, metadata,
                external_step_reader=external_step_reader,
            )
        return None
    return _configure_batch_sampler_sources(
        state, dataset, metadata_fn, metadata, config, batch_sampler, loader_options,
    )


def _finalize_build_state(
        state: _BuildState,
        config: DistributedDatasetConfig,
        dataloader_fingerprint: str,
        effective_pack_fn: Callable[[Sequence[Any], int], Any],
        effective_collate_fn: Callable[[Sequence[Any]], Any],
        uses_default_pack: bool,
        uses_default_collate: bool,
        batch_sampler_fingerprint: str | None,
) -> None:
    """Create the planner/constructor and stable build fingerprint."""
    state.planner = DynamicPackingPlanner(
        data_parallel_size=state.topology.data_parallel_size,
        seq_len=config.seq_len,
        local_batch_size=config.local_batch_size,
        oversized_policy=config.oversized_policy,
        packing_budgets=config.packing_budgets,
        min_balance_gain=config.min_balance_gain,
        model_config=state.model_config,
        cost_model=state.cost_model,
        balancing_algorithm=state.balancing_algorithm,
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
    state.config_fingerprint += ":policies:" + _policy_fingerprint(state.planner)


def _policy_fingerprint(planner: DynamicPackingPlanner) -> str:
    """Include policy versions in native build agreement and checkpoint identity."""
    identities = []
    for policy, field_name in ((planner.cost_model, "model_id"), (planner.balancing_algorithm, "algorithm_id")):
        policy_type = policy if hasattr(policy, "__qualname__") else type(policy)
        identities.append({
            "implementation": f"{policy_type.__module__}.{policy_type.__qualname__}",
            "version": getattr(policy, field_name, None),
        })
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


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
        external_step_source: ExternalStepSource | None = None,
) -> None:
    if not isinstance(config, DistributedDatasetConfig):
        raise ValueError(f"config must be DistributedDatasetConfig, but got {type(config)}.")
    if (external_step_reader is not None or external_step_source is not None) and batch_sampler is not None:
        raise ValueError("external step sources and batch_sampler are mutually exclusive.")
    if external_step_reader is None and external_step_source is None:
        metadata = _infer_dataset_metadata(dataset, metadata_fn, metadata)
    elif metadata is not None or metadata_fn is not None:
        if external_step_reader is not None:
            raise ValueError("external_step_reader cannot be combined with metadata or metadata_fn.")
        if metadata is not None:
            raise ValueError("external_step_source cannot be combined with precomputed metadata.")
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
    effective_pack_fn, effective_collate_fn, uses_default_pack, uses_default_collate = _resolve_constructor_callbacks(
        batch_sampler, pack_fn, collate_fn,
    )
    state.topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
    state.dataset_reader_ranks, state.planner_rank = _resolve_service_ranks(state.topology, config)
    batch_sampler_fingerprint = _configure_step_sources(
        state,
        dataset,
        metadata_fn,
        metadata,
        config,
        loader_options,
        effective_pack_fn,
        effective_collate_fn,
        batch_sampler,
        external_step_reader,
        external_step_source,
    )
    _finalize_build_state(
        state,
        config,
        dataloader_fingerprint,
        effective_pack_fn,
        effective_collate_fn,
        uses_default_pack,
        uses_default_collate,
        batch_sampler_fingerprint,
    )


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
        communication_backend=getattr(config, "communication_backend", "hccl"),
        communication_device=state.communication_device,
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
        device: Any = None,
        batch_sampler: Any = None,
        external_step_reader: Any | None = None,
        external_step_source: ExternalStepSource | None = None,
        model_config: Any = None,
        cost_model: CostModel | None = None,
        balancing_algorithm: BalancingAlgorithm | None = None,
        move_fn: Callable[[Any, Any], Any] | None = None,
        bin_stats_fn: Callable[[Iterable[SampleMetadata]], dict[str, Any]] | None = None,
        max_steps: int | None = None,
) -> DistributedDataLoader | LocalBalancingDataLoader | DatasetDataLoader:
    """Build a sample-balanced distributed DataLoader.

    Online mode without ``batch_sampler`` requires an external step source or
    the legacy ``external_step_reader``. The source emits one complete local
    step of raw-sample bins; HP derives metadata, freezes the sample union,
    balances it across Data Constructors, and routes payloads when necessary.
    Metadata mode requires ``batch_sampler`` to define step/sample boundaries.
    It looks up ``metadata[index]`` before any Dataset read; target constructors
    then directly read their assigned indices from the shared Dataset and skip
    payload A2A. Metadata-only streaming selection is not supported.

    With ``batch_sampler``, native HP sampling owns step selection:
    one sampler yield per DP Constructor fixes one forward/backward round.
    Complete Dataset outputs are balanced without repacking their contents.

    Args:
        dataset: A build_distributed_dataset result supplies metadata, collation
            and field placement itself and yields device-ready microbatches.
            Otherwise, BatchSampler mode requires a shared mapping Dataset
            on Data Constructor ranks. In external-step mode, the Reader owns data
            loading and ``dataset`` may be ``None``. Other ranks may pass the
            same object or ``None``.
        mesh: Named root HyperParallel or native DeviceMesh.
        config: Dynamic packing, service-rank, and worker configuration.
        metadata_fn: Derives metadata from each Dataset output in
            ``batch_sampler`` mode or each raw sample emitted by
            ``external_step_source``.
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
        device: Rank-local training device. Node-local balancing uses
            it for final H2D and for control/payload tensors when
            ``communication_backend="hccl"``. Gloo keeps metadata and raw
            sample communication on the host and defaults to CPU batches;
            pass an explicit NPU/CUDA device to enable H2D with Gloo.
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
            legacy rank-local Reader that must
            expose ``prepare_next_step``, ``metadata``, ``original_metadatas`` (or legacy ``reference_bins``),
            ``selected_payloads``, ``commit``, ``exhausted``, and checkpoint/epoch methods.
            One call to ``prepare_next_step`` supplies exactly one already-selected local
            step. HP preserves that step's union and only rebalances its target
            ranks.
        external_step_source: Preferred source-only API for online mode. It must
            be an iterable whose next value is one complete local step, represented
            as ``local_batch_size`` raw-sample bins on every pure-DP rank.
            HP supplies metadata, payload caching, packing, collation and
            distributed placement through ``metadata_fn``, ``pack_fn`` and
            ``collate_fn``. Optional ``set_epoch`` is forwarded to the source;
            this buffered route does not provide checkpoint/resume.
        model_config: Effective backbone dimensions required when cost_model
            is omitted. The default cost model is constructed from these dimensions.
        cost_model: Optional user workload callback replacing the default.
        balancing_algorithm: Optional assignment and objective policy. Receives
            already-scored samples; Hyper enforces capacities and the gain gate.
        move_fn: H2D field mapping; retain CPU-only metadata here. Called per
            microbatch for local-step sources, or per collated batch for samplers/readers.
        bin_stats_fn: Optional per-bin counters in the local-step rank-zero log.
        max_steps: Local-step step limit, including speculative prefetch.

    Returns:
        Collective iterator yielding constructed local batches.

    Note:
        Checkpoint replay requires a deterministic online stream for a given
        epoch; arbitrary worker-side RNG state is not captured. Metadata and
        Dataset lengths must agree across all Data Constructor ranks. Metadata
        entries must describe deterministic, rank-independent Dataset outputs.
        A DistributedDataset or external_step_source uses pure DP, node-local
        communication and buffered H2D. Every step evaluates a candidate; sample
        exchange occurs only when its objective improves by more than
        min_balance_gain. HCCL is the default backend; Gloo transports control
        and CPU payloads on the host. HCCL transports control and payload
        tensors on the rank-local NPU.
        Checkpoint/resume remains available only on the reader/sampler path.
        Stateful custom policies should expose configuration-versioned model_id
        or algorithm_id attributes for build/checkpoint identity.
    """
    if isinstance(dataset, DistributedDataset):
        if any(value is not None for value in (
                metadata_fn, metadata, pack_fn, collate_fn, move_fn, bin_stats_fn,
                batch_sampler, external_step_source, external_step_reader,
        )) or dataloader_kwargs:
            raise ValueError("Configure source, metadata, collation and field placement on DistributedDataset.")
        device = _resolve_device(device, communication_backend=config.communication_backend)
        loader = build_local_balancing_dataloader(
            dataset, mesh, config,
            metadata_fn=dataset.sample_metadata,
            pack_fn=dataset.pack,
            collate_fn=list,
            model_config=model_config,
            cost_model=cost_model,
            balancing_algorithm=balancing_algorithm,
            device=device,
            move_fn=dataset.move_to_device,
            bin_stats_fn=dataset.summarize if dataset.log_fields else None,
            max_steps=max_steps,
        )
        return DatasetDataLoader(dataset, loader, device)
    if external_step_source is not None:
        if batch_sampler is not None or metadata is not None or external_step_reader is not None:
            raise ValueError("external_step_source cannot be combined with a sampler, metadata sequence or reader.")
        if dataloader_kwargs:
            raise ValueError("Configure source worker options on external_step_source, not the balancing wrapper.")
        return build_local_balancing_dataloader(
            external_step_source,
            mesh,
            config,
            metadata_fn=metadata_fn,
            pack_fn=default_pack_fn if pack_fn is None else pack_fn,
            collate_fn=default_collate_fn if collate_fn is None else collate_fn,
            model_config=model_config,
            cost_model=cost_model,
            balancing_algorithm=balancing_algorithm,
            device=device,
            move_fn=move_fn,
            bin_stats_fn=bin_stats_fn,
            max_steps=max_steps,
        )
    return _build_distributed_dataloader_impl(
        dataset,
        mesh,
        config,
        metadata_fn=metadata_fn,
        metadata=metadata,
        dataloader_kwargs=dataloader_kwargs,
        pack_fn=pack_fn,
        collate_fn=collate_fn,
        communication_device=device,
        batch_sampler=batch_sampler,
        external_step_reader=external_step_reader,
        external_step_source=external_step_source,
        model_config=model_config,
        cost_model=cost_model,
        balancing_algorithm=balancing_algorithm,
        move_fn=move_fn,
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
        external_step_source: ExternalStepSource | None = None,
        model_config: Any = None,
        cost_model: CostModel | None = None,
        balancing_algorithm: BalancingAlgorithm | None = None,
        move_fn: Callable[[Any, Any], Any] | None = None,
) -> DistributedDataLoader:
    if external_step_reader is not None and external_step_source is not None:
        raise ValueError("external_step_reader and external_step_source are mutually exclusive.")
    if communication_device is None:
        communication_device = _resolve_device(communication_backend=getattr(config, "communication_backend", "hccl"))
    metadata_mode = _resolve_metadata_mode(
        metadata_fn,
        metadata,
        external_step_reader,
        external_step_source,
        communication_backend=getattr(config, "communication_backend", "hccl"),
        communication_device=communication_device,
    )
    state = _BuildState(
        metadata_mode=metadata_mode, model_config=model_config,
        cost_model=cost_model, balancing_algorithm=balancing_algorithm,
    )
    device_prefetch = None
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
            external_step_source,
        )
        device = _resolve_device(communication_device)
        device_prefetch = _create_device_prefetcher(device, move_fn)
        state.communication_device = device
    except Exception as exc:  # Every WORLD rank must fail before subgroup creation.
        state.local_error = f"{type(exc).__name__}: {exc}"

    _synchronize_build_state(state, config)
    _require_build_state(state)
    groups = create_data_groups(
        state.topology,
        state.dataset_reader_ranks,
        state.planner_rank,
        cpu_backend=config.communication_backend,
        payload_backend=config.communication_backend,
        communication_device=(
            state.communication_device if config.communication_backend == "hccl" else None
        ),
        enable_payload_exchange=not state.metadata_mode,
        model_device=state.communication_device,
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
            communication_device=(
                state.communication_device if config.communication_backend == "hccl" else None
            ),
        ),
        model_transport=ModelParallelTransport(
            state.topology,
            groups,
            communication_device=(
                state.communication_device if config.communication_backend == "hccl" else None
            ),
        ),
        device_prefetch=device_prefetch if state.topology.is_constructor else None,
        config_fingerprint=state.config_fingerprint,
    )


__all__ = ["DistributedDatasetConfig", "build_distributed_dataloader"]
