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
"""Public builder for sample-balanced distributed dynamic packing.

Source modes and buffering
--------------------------
Online mode without ``batch_sampler`` requires ``external_step_source``.
The source emits one complete local step of raw-sample bins. HP derives
metadata, freezes the sample union,
balances it across Data Constructors, and routes payloads when necessary.
Metadata mode requires ``batch_sampler`` to define step/sample boundaries.
It looks up ``metadata[index]`` before any Dataset read; target constructors
then directly read their assigned indices from the shared Dataset and skip
payload A2A. Metadata-only streaming selection is not supported.

With ``batch_sampler``, native HP sampling owns step selection:
``balance_window_size`` consecutive sampler yields per DP Constructor are
planned together, stopping at optimizer and epoch boundaries. Each iterator
yield still returns exactly one forward/backward round. The current window
and at most one future window are double-buffered on Host. Device staging
independently retains up to ``device_prefetch_depth`` future micro-batches;
model-parallel input broadcast runs when the current batch is consumed.
Checkpoints advance only at complete optimizer-step boundaries.
Dataset outputs stay whole by default. Online ``split_fn`` decomposes each
selected output into independent sequences; ``pack_fn`` rebuilds the planned
bins without changing the original sampler window's sample membership.
Set ``config.repack_sequences=True`` to use built-in text callbacks instead
of supplying split_fn/pack_fn. ``config.pad_token_id`` controls source PAD
trimming and new PAD; ``config.log_balance`` enables balancing reports.

Checkpoint and distributed contracts
-----------------------------------
Native sampler checkpoint replay requires a deterministic Dataset stream
for a given epoch; arbitrary worker-side RNG state is not captured. Metadata and
Dataset lengths must agree across all Data Constructor ranks; the caller
must ensure this consistency because the builder only validates local inputs. Metadata
entries must describe deterministic, rank-independent Dataset outputs.
An external_step_source uses pure DP, node-local
communication and buffered H2D. Every step evaluates a candidate; sample
exchange occurs only when its objective improves by more than
min_balance_gain. HCCL is the default backend; Gloo transports control
and CPU payloads on the host. HCCL transports control and payload
tensors on the rank-local NPU.
Checkpoint/resume remains available only on the sampler path, at complete
optimizer-step boundaries. Save training state after the optimizer update;
an incomplete step retains its starting cursor and replays in full.
The caller must keep topology, configuration and custom policies
consistent across ranks and compatible with restored checkpoints.
"""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from multiprocessing.context import BaseContext
from typing import Any, Literal

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.batch_sampler import (
    BatchSamplerReader,
    preserve_sample,
    validate_native_sampler,
)
from hyper_parallel.distributed_data.data_constructor import (
    PackingDataConstructor,
    default_collate_fn,
    default_pack_fn,
)
from hyper_parallel.distributed_data.device_prefetch import _create_device_prefetcher, _resolve_device
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.balancing_algorithm import BalancingAlgorithm
from hyper_parallel.distributed_data.cost_model import CausalAttentionCostModel, CostModel
from hyper_parallel.distributed_data.model_batch import TensorFieldSpec
from hyper_parallel.distributed_data.indexed_text import (
    _validate_pad_token_id,
    collate_indexed_text_sequences,
    pack_indexed_text_samples,
    split_packed_text_sample,
)
from hyper_parallel.distributed_data.packed_balancing import LocalBalancingDataLoader, build_local_balancing_dataloader
from hyper_parallel.distributed_data.external_step import ExternalStepSource
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner, OversizedPolicy
from hyper_parallel.distributed_data.schema import PackingConstraints, SampleMetadata
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.dataset_reader import _validate_worker_options
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    create_data_groups,
)

_DEFAULT_DATALOADER_KWARGS = {
    "num_workers": 0,
    "pin_memory": False,
    "prefetch_factor": None,
    "persistent_workers": False,
}
_OPTIONAL_DATALOADER_KWARGS = frozenset({
    "in_order",
    "multiprocessing_context",
    "pin_memory_device",
    "timeout",
    "worker_init_fn",
})
_SUPPORTED_DATALOADER_KWARGS = frozenset(_DEFAULT_DATALOADER_KWARGS) | _OPTIONAL_DATALOADER_KWARGS
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
    """Configure sample planning, dynamic packing, and data-plane communication.

    ``seq_len`` and ``local_batch_size`` are the only required sizing inputs.
    One iterator yield contains ``local_batch_size`` packed sequences on each
    DP rank. Native sampler global batch size defines optimizer boundaries.

    Args:
        seq_len: Maximum token count in one non-oversized packed sequence.
        local_batch_size: Packed sequences constructed per DP rank and yield.
        balance_window_size: Maximum native sampler micro-batches planned together.
            Native prefetch buffers the current window and at most one future window,
            each clipped at the optimizer-step boundary.
            Defaults to one. Windows stop at optimizer and epoch boundaries;
            gradient accumulation is inferred from the native sampler's global
            batch size. External-step sources support only one.
        device_prefetch_depth: Future native NPU/CUDA micro-batches copied ahead
            on Constructors. Model peers receive each batch only at delivery.
            Defaults to one. Zero stages only on
            demand while retaining Host window prefetch. Must be a non-negative
            integer and identical across ranks. Independent of balance_window_size;
            ignored on CPU. External-step sources support only the default value.
        dp_dim_names: Named mesh dimensions that define DP coordinates.
        balance_group_size: Optional number of DP ranks in each independent
            local-balancing group. ``None`` preserves launcher/node grouping.
            This applies only to external-step
            loading; native ``batch_sampler`` loading remains global.
        dataset_reader_ranks: Optional Dataset Reader ranks. They read raw
            samples online or metadata only in metadata mode. Defaults to the
            Data Constructor ranks.
        dataset_already_sharded: Legacy sharding configuration flag.
            Native BatchSampler requires False because it owns DP slicing;
            external sources manage their own partitioning.
        planner_rank: Optional centralized Planner rank. Defaults to the lowest
            Data Constructor rank.
        oversized_policy: ``error`` by default; ``single`` explicitly permits
            one oversized sample to occupy a bin alone.
        drop_last: Whether to drop a tail with fewer than one sample per global
            packing bin. Only ``True`` is supported in this first version.
        seed: DataLoader worker seed. Sampling order is configured on the sampler
            or external step producer.
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
        metadata_mode: Whether metadata is precomputed and available by Dataset
            index. ``False`` means the Dataset Reader loads samples and calls
            ``metadata_fn`` during native sampler planning.
        broadcast_plan: Whether one Planner computes and broadcasts each plan.
            Defaults to ``True``. If ``False``, every data-plane rank computes
            the plan from gathered metadata. Configuration and deterministic
            cost/assignment policies must produce identical plans across ranks.
        log_balance: Whether to emit INFO-level balancing reports. Defaults to
            ``False``; enabling INFO logging alone does not enable these reports.
        repack_sequences: Use built-in text splitting and packing on the native
            online sampler path. Defaults to ``False``. Records must contain
            input_ids, pre-shifted labels and explicit cu_seq_lens. Cannot be
            combined with custom split_fn or pack_fn. Unless overridden, use
            the text collator and causal Attention square cost including new PAD.
        pad_token_id: Optional non-negative source PAD ID for built-in repacking.
            Only trailing matching tokens with labels=-100 are removed. The same
            ID fills new PAD. ``None`` preserves source tokens and fills with 0.
            Has no effect when repack_sequences is false.
    """

    seq_len: int
    local_batch_size: int
    dp_dim_names: tuple[str, ...] | None = None
    dataset_reader_ranks: tuple[int, ...] | None = None
    dataset_already_sharded: bool = False
    planner_rank: int | None = None
    oversized_policy: OversizedPolicy = "error"
    drop_last: bool = True
    seed: int = 1234
    min_balance_gain: float = 0.0
    packing_budgets: dict[str, float] | None = None
    communication_backend: Literal["gloo", "hccl"] = "hccl"
    balance_group_size: int | None = None
    metadata_mode: bool = False
    broadcast_plan: bool = True
    balance_window_size: int = 1
    log_balance: bool = False
    repack_sequences: bool = False
    pad_token_id: int | None = None
    device_prefetch_depth: int = 1

    def __post_init__(self) -> None:
        """Validate topology-independent configuration boundaries."""
        self._validate_integer_fields()
        self._validate_policy_and_flags()
        if self.pad_token_id is not None:
            _validate_pad_token_id(self.pad_token_id)
        if self.repack_sequences and self.metadata_mode:
            raise ValueError("repack_sequences requires online metadata (metadata_mode=False).")
        self._validate_rank_tuple(self.dataset_reader_ranks, "dataset_reader_ranks")
        self._validate_name_tuple(self.dp_dim_names, "dp_dim_names")
        self._validate_planner_rank()
        PackingConstraints(self.seq_len, self.oversized_policy, self.packing_budgets)

    def _validate_integer_fields(self) -> None:
        """Validate window sizes, batch dimensions and non-negative integer options."""
        for name in ("seq_len", "local_batch_size", "balance_window_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        for name in ("seed", "device_prefetch_depth"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, but got {value!r}.")
        if (
            self.balance_group_size is not None
            and (
                not isinstance(self.balance_group_size, int)
                or isinstance(self.balance_group_size, bool)
                or self.balance_group_size < 1
            )
        ):
            raise ValueError("balance_group_size must be a positive integer or None.")

    def _validate_policy_and_flags(self) -> None:
        """Validate policy names, boolean flags and the relative-gain threshold."""
        if self.oversized_policy not in ("error", "single"):
            raise ValueError("oversized_policy must be 'error' or 'single'.")
        for name in (
                "dataset_already_sharded",
                "drop_last",
                "metadata_mode",
                "broadcast_plan",
                "log_balance",
                "repack_sequences",
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
    """Resolve Reader and planner ownership within the root mesh."""
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
        dataloader_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve validated DataLoader execution options."""
    if dataloader_kwargs is None:
        supplied_options = {}
    elif not isinstance(dataloader_kwargs, Mapping):
        raise ValueError("dataloader_kwargs must be a mapping or None.")
    else:
        try:
            supplied_options = dict(dataloader_kwargs)
        except Exception as exc:
            raise ValueError(f"dataloader_kwargs cannot be copied: {exc}") from exc
    _validate_dataloader_option_keys(supplied_options)
    options = dict(_DEFAULT_DATALOADER_KWARGS)
    options.update(supplied_options)
    _validate_worker_options(options)
    _validate_dataloader_timeout(options)
    worker_init_fn = options.get("worker_init_fn")
    if worker_init_fn is not None and not callable(worker_init_fn):
        raise ValueError("dataloader_kwargs worker_init_fn must be callable or None.")
    _validate_multiprocessing_context(options)
    pin_memory_device = options.get("pin_memory_device", "")
    if not isinstance(pin_memory_device, str):
        raise ValueError("dataloader_kwargs pin_memory_device must be a string.")
    return options


def _validate_dataloader_option_keys(supplied_options: Mapping[str, Any]) -> None:
    """Reject unsupported keys and overrides of internally managed DataLoader options."""
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


def _validate_dataloader_timeout(options: Mapping[str, Any]) -> None:
    """Require a finite timeout compatible with the configured worker count."""
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


def _validate_multiprocessing_context(options: Mapping[str, Any]) -> None:
    """Check that the worker context uses a supported multiprocessing start method."""
    multiprocessing_context = options.get("multiprocessing_context")
    if multiprocessing_context is None:
        return
    if options["num_workers"] == 0:
        raise ValueError("dataloader_kwargs multiprocessing_context requires num_workers > 0.")
    if not isinstance(multiprocessing_context, (str, BaseContext)):
        raise ValueError("dataloader_kwargs multiprocessing_context is invalid.")
    start_method = multiprocessing_context
    if isinstance(multiprocessing_context, BaseContext):
        try:
            start_method = multiprocessing_context.get_start_method()
        except Exception as exc:
            raise ValueError(
                f"dataloader_kwargs multiprocessing_context cannot report its start method: {exc}"
            ) from exc
    if start_method not in torch.multiprocessing.get_all_start_methods():
        raise ValueError(f"dataloader_kwargs multiprocessing_context={start_method!r} is not supported.")


def _normalize_communication_device(communication_device: Any) -> torch.device | None:
    """Normalize a rank-local A2A device."""
    if communication_device is None:
        return None
    try:
        return torch.device(communication_device)
    except Exception as exc:
        raise ValueError(f"communication_device is invalid: {communication_device!r}.") from exc


class _BuildState:
    """Validated rank-local components needed to assemble a distributed DataLoader."""

    def __init__(
            self,
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
            *,
            split_fn: Callable[[Any], Sequence[Any]] | None = None,
            model_config: Any = None,
            cost_model: CostModel | None = None,
            balancing_algorithm: BalancingAlgorithm | None = None,
    ) -> None:
        """Validate inputs and construct all rank-local components.

        Args:
            dataset: Mapping Dataset on each Reader rank.
            mesh: Named root device mesh.
            config: Explicit loading mode, topology and packing options.
            metadata_fn: Derives metadata from loaded samples in online mode.
            metadata: Indexed precomputed metadata in metadata mode.
            dataloader_kwargs: Rank-local sample loading options.
            pack_fn: Optional packed-sequence constructor.
            collate_fn: Optional local-batch collator.
            communication_device: Rank-local control and payload device.
            batch_sampler: Native sampler defining each local batch.
            split_fn: Optional decomposition of each selected record into independent sequences.
            model_config: Backbone dimensions for the default cost model.
            cost_model: Optional workload estimator.
            balancing_algorithm: Optional assignment and objective policy.
        """
        if not isinstance(config, DistributedDatasetConfig):
            raise ValueError(f"config must be DistributedDatasetConfig, but got {type(config)}.")
        _validate_builder_callbacks(metadata_fn, metadata, pack_fn, collate_fn)
        split_fn, pack_fn, collate_fn = _resolve_repacking_callbacks(
            config, batch_sampler, split_fn, pack_fn, collate_fn,
        )
        if split_fn is not None and (not callable(split_fn) or config.metadata_mode):
            raise ValueError("split_fn requires online metadata and must be callable.")
        normalized_options = _normalize_dataloader_kwargs(dataloader_kwargs)
        loader_options = {name: normalized_options[name] for name in _DEFAULT_DATALOADER_KWARGS}
        loader_options["dataloader_kwargs"] = {
            name: value
            for name, value in normalized_options.items()
            if name not in _DEFAULT_DATALOADER_KWARGS
        }
        self.communication_device = _normalize_communication_device(communication_device)
        if batch_sampler is None:
            if config.metadata_mode:
                raise ValueError("Metadata mode requires batch_sampler to define step/sample boundaries.")
            raise ValueError("Online mode requires batch_sampler or external_step_source to define complete steps.")
        effective_pack_fn, effective_collate_fn = _resolve_constructor_callbacks(
            pack_fn, collate_fn, split_fn,
        )
        self.topology = DataTopology.from_mesh(mesh, dp_dim_names=config.dp_dim_names)
        self.dataset_reader_ranks, self.planner_rank = _resolve_service_ranks(self.topology, config)
        self.dataset_reader: BatchSamplerReader | None = None
        self.metadata_reader: BatchSamplerReader | None = None
        self.direct_sample_loader: PlannedSampleLoader | None = None
        if config.metadata_mode:
            if metadata_fn is not None:
                raise ValueError("metadata_mode=True requires precomputed metadata, not metadata_fn.")
            if metadata is None:
                raise ValueError("metadata_mode=True requires metadata.")
        elif metadata is not None:
            raise ValueError("metadata_mode=False requires metadata_fn, not precomputed metadata.")
        elif metadata_fn is None:
            raise ValueError("metadata_mode=False requires metadata_fn.")
        self._configure_batch_sampler_sources(
            dataset, metadata_fn, metadata, config, batch_sampler, loader_options, split_fn,
        )
        self.planner, self.constructor = self._build_components(
            config,
            effective_pack_fn,
            effective_collate_fn,
            model_config,
            cost_model,
            balancing_algorithm,
        )

    @property
    def is_reader(self) -> bool:
        """Return whether this rank owns a Dataset Reader."""
        return self.topology.global_rank in self.dataset_reader_ranks

    def _configure_batch_sampler_sources(
            self,
            dataset: Any,
            metadata_fn: Callable[[Any], SampleMetadata] | None,
            metadata: Sequence[SampleMetadata] | None,
            config: DistributedDatasetConfig,
            batch_sampler: Any,
            loader_options: dict[str, Any],
            split_fn: Callable[[Any], Sequence[Any]] | None,
    ) -> None:
        """Use native DP sampler owners as Readers without applying a second stride."""
        if self.dataset_reader_ranks != self.topology.constructor_ranks:
            raise ValueError("Native batch_sampler currently requires Dataset Readers on the Data Constructor ranks.")
        if config.dataset_already_sharded:
            raise ValueError(
                "Native batch_sampler owns DP slicing; disable dataset_already_sharded."
            )
        validate_native_sampler(
            batch_sampler, data_rank=self.topology.data_rank,
            dp_size=self.topology.data_parallel_size, local_batch_size=config.local_batch_size,
        )
        if not self.is_reader:
            return
        if dataset is None:
            raise ValueError("Native batch_sampler requires a mapping Dataset on each Data Constructor.")
        sample_loader = PlannedSampleLoader(dataset, seed=config.seed, **loader_options)
        sample_loader.set_epoch(batch_sampler.epoch)
        if config.metadata_mode:
            if not callable(getattr(metadata, "__getitem__", None)) or len(metadata) != len(dataset):
                raise ValueError("Native batch_sampler metadata must align with the mapping Dataset.")
            self.direct_sample_loader = sample_loader
        reader = BatchSamplerReader(
            batch_sampler, reader_rank=self.topology.global_rank,
            metadata=metadata, metadata_fn=metadata_fn,
            sample_loader=None if config.metadata_mode else sample_loader,
            split_fn=split_fn,
        )
        if config.metadata_mode:
            self.metadata_reader = reader
        else:
            self.dataset_reader = reader

    def _build_components(
            self,
            config: DistributedDatasetConfig,
            effective_pack_fn: Callable[[Sequence[Any], int], Any],
            effective_collate_fn: Callable[[Sequence[Any]], Any],
            model_config: Any,
            cost_model: CostModel | None,
            balancing_algorithm: BalancingAlgorithm | None,
    ) -> tuple[DynamicPackingPlanner, PackingDataConstructor]:
        """Create the planner and data constructor."""
        if cost_model is None and config.repack_sequences:
            cost_model = CausalAttentionCostModel()
        planner = DynamicPackingPlanner(
            data_parallel_size=self.topology.data_parallel_size,
            seq_len=config.seq_len,
            local_batch_size=config.local_batch_size,
            oversized_policy=config.oversized_policy,
            packing_budgets=config.packing_budgets,
            min_balance_gain=config.min_balance_gain,
            model_config=model_config,
            cost_model=cost_model,
            balancing_algorithm=balancing_algorithm,
        )
        return planner, PackingDataConstructor(effective_pack_fn, effective_collate_fn, seq_len=config.seq_len)


def _validate_builder_callbacks(
        metadata_fn: Callable[[Any], SampleMetadata] | None,
        metadata: Sequence[SampleMetadata] | None,
        pack_fn: Callable[[Sequence[Any], int], Any] | None,
        collate_fn: Callable[[Sequence[Any]], Any] | None,
) -> None:
    """Validate callback types and mutually exclusive metadata providers."""
    if metadata_fn is not None and not callable(metadata_fn):
        raise ValueError("metadata_fn must be callable or None.")
    if metadata_fn is not None and metadata is not None:
        raise ValueError("Provide either metadata or metadata_fn, but not both.")
    if pack_fn is not None and not callable(pack_fn):
        raise ValueError("pack_fn must be callable or None.")
    if collate_fn is not None and not callable(collate_fn):
        raise ValueError("collate_fn must be callable or None.")


def _resolve_repacking_callbacks(
        config: DistributedDatasetConfig,
        batch_sampler: Any,
        split_fn: Callable[[Any], Sequence[Any]] | None,
        pack_fn: Callable[[Sequence[Any], int], Any] | None,
        collate_fn: Callable[[Sequence[Any]], Any] | None,
) -> tuple[
    Callable[[Any], Sequence[Any]] | None,
    Callable[[Sequence[Any], int], Any] | None,
    Callable[[Sequence[Any]], Any] | None,
]:
    """Select built-in text callbacks without mixing packing contracts."""
    if not config.repack_sequences:
        return split_fn, pack_fn, collate_fn
    if batch_sampler is None:
        raise ValueError("repack_sequences requires a native batch_sampler.")
    if split_fn is not None or pack_fn is not None:
        raise ValueError("repack_sequences cannot be combined with custom split_fn or pack_fn.")
    return (
        partial(split_packed_text_sample, pad_token_id=config.pad_token_id),
        partial(pack_indexed_text_samples, pad_token_id=0 if config.pad_token_id is None else config.pad_token_id),
        collate_indexed_text_sequences if collate_fn is None else collate_fn,
    )


def _resolve_constructor_callbacks(
        pack_fn: Callable[[Sequence[Any], int], Any] | None,
        collate_fn: Callable[[Sequence[Any]], Any] | None,
        split_fn: Callable[[Any], Sequence[Any]] | None = None,
) -> tuple[Callable[[Sequence[Any], int], Any], Callable[[Sequence[Any]], Any]]:
    """Resolve constructor callbacks for a native BatchSampler source."""
    if split_fn is not None and pack_fn is None:
        raise ValueError("Native split_fn requires pack_fn to rebuild the planned sequences.")
    if split_fn is None and pack_fn is not None:
        raise ValueError("Native batch_sampler preserves Dataset outputs; pack_fn must be omitted.")
    effective_pack_fn = preserve_sample if split_fn is None else pack_fn
    effective_collate_fn = default_collate_fn if collate_fn is None else collate_fn
    return effective_pack_fn, effective_collate_fn


def build_distributed_dataloader(
        dataset: Any | None,
        mesh: Any,
        config: DistributedDatasetConfig,
        *,
        metadata_fn: Callable[[Any], SampleMetadata] | None = None,
        metadata: Sequence[SampleMetadata] | None = None,
        dataloader_kwargs: Mapping[str, Any] | None = None,
        pack_fn: Callable[[Sequence[Any], int], Any] | None = None,
        split_fn: Callable[[Any], Sequence[Any]] | None = None,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
        device: Any = None,
        model_batch_spec: Any = None,
        batch_sampler: Any = None,
        external_step_source: ExternalStepSource | None = None,
        model_config: Any = None,
        cost_model: CostModel | None = None,
        balancing_algorithm: BalancingAlgorithm | None = None,
) -> DistributedDataLoader | LocalBalancingDataLoader:
    """Build a sample-balanced loader using the source and checkpoint contracts above.

    Args:
        dataset: Shared mapping Dataset on native Data Constructors. External sources
            own loading and may pass None; other ranks may also pass None.
        mesh: Named root HyperParallel or native DeviceMesh.
        config: Packing, repacking, logging, service-rank and communication options.
        metadata_fn: Derive metadata from each native Dataset output or external raw
            sample. With splitting, receives each independent sequence instead.
        metadata: Explicit indexable metadata on native Dataset Reader ranks in metadata
            mode. Must implement __len__ and __getitem__; metadata[index] must describe
            the corresponding deterministic, rank-independent dataset[index].
        dataloader_kwargs: Native worker options: num_workers=0, pin_memory=False,
            prefetch_factor=None, persistent_workers=False by default. Sampling,
            batching, shuffling and DataLoader collation cannot be overridden.
        pack_fn: Build one packed sequence; the external default preserves a bin as a
            raw-sample tuple. Omit when config.repack_sequences selects text packing.
        split_fn: Online native callback returning independent sequences in original
            order. Requires pack_fn. Each original record must fit one bin and its
            sequences retain their fallback grouping. Preserve tokens, targets and
            attention isolation; contextual slices are not independent. Omit with
            config.repack_sequences; unsupported in metadata mode.
        collate_fn: Collate local_batch_size planned bins. Defaults to text collation
            with config.repack_sequences, otherwise a tuple; an explicit callback wins.
        device: Rank-local training device. HCCL uses its NPU for control, payload and
            final H2D. Gloo communicates on Host and defaults to CPU; pass an explicit
            NPU/CUDA device to enable H2D with Gloo.
        model_batch_spec: Identical layout on every MP rank: nested dict/list/tuple
            leaves are TensorFieldSpec or static scalars. Dynamic values must be tensors;
            fields and dtypes are not broadcast. Defaults to collate_fn.model_batch_spec
            or the built-in text layout. Required for custom MP batches, unnecessary for DP.
        batch_sampler: Native HP BatchSampler supplied on every rank; only Constructors
            advance it. Readers must coincide with Constructors and drop_last must be
            true. Dataset outputs stay whole unless split_fn/pack_fn or online text
            repacking is enabled. Metadata indexes outputs, not underlying documents.
            Checkpoint via this loader, not the sampler's speculative cursor.
        external_step_source: Online pure-DP iterable emitting one complete local step
            of local_batch_size raw-sample bins. Optional set_epoch is forwarded;
            exhaustion ends iteration. The caller owns recovery and step limits.
        model_config: Backbone dimensions for the default cost model when repacking
            is disabled and cost_model is omitted.
        cost_model: Optional workload callback. Text repacking defaults to
            CausalAttentionCostModel including new PAD; otherwise DefaultCostModel.
        balancing_algorithm: Optional assignment/objective policy receiving scored
            samples. HyperParallel enforces capacity and the gain gate.

    Returns:
        Collective iterator yielding one constructed local batch per micro-batch.
    """
    if external_step_source is not None:
        if model_batch_spec is not None:
            raise ValueError("model_batch_spec is only supported with a native batch_sampler.")
        if split_fn is not None:
            raise ValueError("split_fn is only supported with a native batch_sampler.")
        if config.metadata_mode:
            raise ValueError("external_step_source requires metadata_mode=False for online metadata.")
        if batch_sampler is not None or metadata is not None:
            raise ValueError("external_step_source cannot be combined with a sampler or metadata sequence.")
        if dataloader_kwargs:
            raise ValueError("Configure source worker options on external_step_source, not the balancing wrapper.")
        return build_local_balancing_dataloader(
            external_step_source, mesh, config, metadata_fn=metadata_fn,
            pack_fn=default_pack_fn if pack_fn is None else pack_fn,
            collate_fn=default_collate_fn if collate_fn is None else collate_fn,
            model_config=model_config, cost_model=cost_model,
            balancing_algorithm=balancing_algorithm, device=device,
        )
    return _build_distributed_dataloader_impl(
        dataset, mesh, config, metadata_fn=metadata_fn, metadata=metadata,
        dataloader_kwargs=dataloader_kwargs, pack_fn=pack_fn, collate_fn=collate_fn,
        split_fn=split_fn, communication_device=device, model_batch_spec=model_batch_spec,
        batch_sampler=batch_sampler, model_config=model_config,
        cost_model=cost_model, balancing_algorithm=balancing_algorithm,
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
        model_batch_spec: Any = None,
        batch_sampler: Any = None,
        split_fn: Callable[[Any], Sequence[Any]] | None = None,
        model_config: Any = None,
        cost_model: CostModel | None = None,
        balancing_algorithm: BalancingAlgorithm | None = None,
) -> DistributedDataLoader:
    """Build native sampler readers, planning, transport and device staging for this rank."""
    if communication_device is None:
        communication_device = _resolve_device(communication_backend=getattr(config, "communication_backend", "hccl"))
    if getattr(config, "balance_group_size", None) is not None:
        raise ValueError("balance_group_size is only supported for local balancing paths.")
    state = _BuildState(
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
        split_fn=split_fn,
        model_config=model_config,
        cost_model=cost_model,
        balancing_algorithm=balancing_algorithm,
    )
    device = state.communication_device
    # Validate device/stream support on every rank before any group is created.
    device_prefetch = _create_device_prefetcher(device)
    communication_device = device if config.communication_backend == "hccl" else None
    if model_batch_spec is None:
        model_batch_spec = getattr(collate_fn, "model_batch_spec", None)
    if model_batch_spec is None and (
            collate_fn is collate_indexed_text_sequences or (config.repack_sequences and collate_fn is None)
    ):
        shape = (config.local_batch_size, config.seq_len)
        model_batch_spec = {
            "input_ids": TensorFieldSpec(shape, torch.int64),
            "labels": TensorFieldSpec(shape, torch.int64),
            "cu_seq_lens": TensorFieldSpec((None,), torch.int32),
        }
    groups = create_data_groups(
        state.topology,
        state.dataset_reader_ranks,
        state.planner_rank,
        cpu_backend=config.communication_backend,
        payload_backend=config.communication_backend,
        communication_device=communication_device,
        enable_payload_exchange=not config.metadata_mode,
        model_device=device,
    )

    accumulation_steps = batch_sampler.global_batch_size // batch_sampler.global_micro_batch_size
    return DistributedDataLoader(
        accumulation_steps=accumulation_steps,
        balance_window_size=config.balance_window_size,
        device_prefetch_depth=config.device_prefetch_depth,
        device=device,
        initial_epoch=batch_sampler.epoch,
        initial_consumed_samples=batch_sampler.consumed_samples,
        topology=state.topology,
        dataset_reader_ranks=state.dataset_reader_ranks,
        dataset_reader=state.dataset_reader,
        metadata_reader=state.metadata_reader,
        direct_sample_loader=state.direct_sample_loader,
        metadata_mode=config.metadata_mode,
        broadcast_plan=config.broadcast_plan,
        log_balance=config.log_balance,
        planner=state.planner,
        data_constructor=state.constructor,
        data_plane=DataPlaneTransport(
            groups,
            state.topology.global_rank,
            communication_device=communication_device,
        ),
        model_transport=ModelParallelTransport(
            state.topology,
            groups,
            model_batch_spec=model_batch_spec,
            device=device,
            communication_device=communication_device,
        ),
        device_prefetch=device_prefetch if state.topology.is_constructor else None,
    )


__all__ = ["DistributedDatasetConfig", "build_distributed_dataloader"]
