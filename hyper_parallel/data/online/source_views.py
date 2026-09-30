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
"""Plan, adapt and replay finite indexed sources as canonical Online samples."""

from __future__ import annotations

import math
import operator
import random
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from heapq import heapify, heappop, heappush
from itertools import chain, zip_longest
from typing import Any, Literal, Protocol, TypeAlias

import torch
from torch.utils import data as torch_data
from torch.utils.data import IterableDataset

from hyper_parallel.data.parallel import DataLoaderParallelContext

_DEFAULT_MAX_BALANCE_GROUP_SIZE = 262_144
_ReadMetric: TypeAlias = str | Mapping[str, float]
AccessMode: TypeAlias = Literal["mapping", "iterable"]
SampleAdapter: TypeAlias = Callable[[Any], Mapping[str, Any]]
SOURCE_INFO_KEY = "__source_info__"

__all__ = [
    "IndexedSource", "IndexedMappingView", "IndexedIterableView", "IndexedBlendMappingView",
    "build_indexed_view", "resolve_read_balance", "validate_indexed_dataloader",
    "AccessMode", "SampleAdapter", "SOURCE_INFO_KEY", "SourceInfo", "IndexedAccessContext",
]


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Stable provenance attached to one raw source record."""

    provider: str
    sample_key: str | None = None
    source_path: str | None = None
    shard_path: str | None = None


@dataclass(frozen=True)
class IndexedAccessContext:
    """Trainer-derived plan used only while constructing indexed access views."""

    random_seed: int = 42
    split_sizes: tuple[int, int, int] | None = None
    dataloader_context: DataLoaderParallelContext | None = None
    micro_batch_size: int = 1


class IndexedSource(Protocol):
    """Finite raw records addressable by a stable logical integer index."""

    def __len__(self) -> int:
        """Return the number of records."""

    def __getitem__(self, index: int) -> Any:
        """Read one raw record by logical index."""


class _ReadCostProvider(Protocol):
    """Provide one metadata-only cost without reading sample payloads."""

    def cost_for_index(self, index: int, metric: _ReadMetric) -> float:
        """Return the estimated read cost for one logical source index."""


def resolve_read_balance(options: Mapping[str, Any]) -> tuple[_ReadMetric, str]:
    """Resolve the optional generic read-balance configuration.

    ``read_balance`` groups metric and strategy. Flat ``balance_by`` and
    ``balance_policy`` are accepted as aliases; ``greedy`` and ``lpt`` select
    the same longest-processing-time heuristic.
    """
    configured = options.get("read_balance")
    if configured is None:
        return options.get("balance_by", "sample"), str(options.get("balance_policy", "none"))
    if configured is False:
        return "sample", "none"
    if isinstance(configured, str):
        if configured.lower() == "none":
            return "sample", "none"
        return configured, "greedy"
    if not isinstance(configured, Mapping):
        raise TypeError("read_balance must be a string, mapping, false, or omitted")
    metric = configured.get("metric", configured.get("by", "sample"))
    strategy = str(configured.get("strategy", configured.get("policy", "greedy")))
    return metric, strategy


def validate_indexed_dataloader(
    dataset: Any, *, sampler_type: str = "single", data_rearrange_map: Any = None,
    dp_world_size: int | None = None, micro_batch_size: int | None = None,
) -> tuple[IndexedMappingView | IndexedIterableView, ...]:
    """Validate new indexed views through the existing Online wrapper graph.

    Returns:
        Indexed views for further DataLoader lifecycle validation. Existing
        file and Hub sources produce an empty tuple and retain their behavior.
    """
    pending = [(dataset, False)]
    visited: set[int] = set()
    views = []
    while pending:
        current, blended = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, IndexedMappingView):
            current.validate_sampler(
                sampler_type=sampler_type, data_rearrange_map=data_rearrange_map,
                dp_world_size=dp_world_size, micro_batch_size=micro_batch_size,
            )
            if blended and isinstance(current.indices, _BalancedIndexPlan):
                raise ValueError("Mapping read_balance cannot be applied before a multi-source blend")
            views.append(current)
        elif isinstance(current, IndexedIterableView):
            views.append(current)
        for attribute in ("source_dataset", "source", "dataset"):
            child = getattr(current, attribute, None)
            if child is not None:
                pending.append((child, blended))
        children = getattr(current, "source_datasets", None)
        if isinstance(children, (list, tuple)):
            pending.extend((child, True) for child in children)
    return tuple(views)


def build_indexed_view(
    source: IndexedSource,
    *,
    access_mode: AccessMode,
    context: IndexedAccessContext,
    data_config: Mapping[str, Any],
    split: str = "train",
    sample_adapter: SampleAdapter | None = None,
    is_valid_sample: Callable[[Mapping[str, Any]], bool] | None = None,
) -> Any:
    """Apply shared access, scheduling and adaptation to an indexed raw source.

    Mapping views retain global indices for HP's existing batch sampler;
    Iterable views own DP/worker sharding. Neither path changes model semantics.
    The source needs ``cost_for_index`` only when metadata scheduling is enabled.
    ``is_valid_sample`` checks decoded samples; failures raise unless the caller
    enables ``filter_samples`` to discard them. Mapping filtering scans payloads.
    """
    balance_by, balance_policy = resolve_read_balance(data_config)
    filter_samples = data_config.get("filter_samples", False)
    if not isinstance(filter_samples, bool):
        raise TypeError("filter_samples must be a boolean")
    if balance_policy != "none" and filter_samples and is_valid_sample is not None:
        raise ValueError(
            "Read scheduling cannot follow content-based sample filtering; "
            "put selection in the prepared metadata or disable read scheduling"
        )
    dp_world = max(1, int(getattr(context.dataloader_context, "dp_world_size", 1)))
    micro_batch = max(1, int(context.micro_batch_size))
    shuffle = bool(data_config.get("shuffle", False))
    if access_mode == "mapping":
        split_index = ("train", "valid", "test").index(split)
        size = context.split_sizes[split_index] if context.split_sizes is not None else None
        view = IndexedMappingView(
            source,
            size=None if filter_samples and is_valid_sample is not None else size,
            seed=context.random_seed,
            shuffle=shuffle,
            balance_by=balance_by,
            balance_policy=balance_policy,
            balance_slots=dp_world,
            balance_group_size=dp_world * micro_batch,
        )
        adapted = _AdaptedMappingView(view, sample_adapter, is_valid_sample, filter_samples=filter_samples)
        if size is not None and filter_samples and is_valid_sample is not None:
            return IndexedMappingView(adapted, size=size, seed=context.random_seed)
        return adapted
    if access_mode == "iterable":
        view = IndexedIterableView(
            source,
            context=context.dataloader_context,
            seed=context.random_seed,
            shuffle=shuffle,
            repeat=bool(data_config.get("repeat", False)),
            output_index_for_resume=bool(data_config.get("output_index_for_resume", False)),
            balance_by=balance_by,
            balance_policy=balance_policy,
            balance_group_size=data_config.get("balance_group_size"),
            max_balance_group_size=int(data_config.get("max_balance_group_size", _DEFAULT_MAX_BALANCE_GROUP_SIZE)),
        )
        return _AdaptedIterableView(view, sample_adapter, is_valid_sample, filter_samples=filter_samples)
    raise ValueError("access_mode must be 'mapping' or 'iterable'")


class _IndexPermutation(Sequence[int]):
    """Constant-memory, seeded Feistel permutation of a finite index space.

    Cycle walking restricts a six-round permutation to the source length. The
    enclosing power-of-four domain is less than four times the source length,
    so a complete traversal averages fewer than four domain permutations per
    sample. This is a deterministic shuffle, not a cryptographic primitive.
    """

    def __init__(self, length: int, seed: int, *, shuffle: bool) -> None:
        """Store the finite permutation parameters."""
        self.length = int(length)
        if self.length < 0:
            raise ValueError("Indexed source permutation length must be non-negative")
        self.shuffle = bool(shuffle and self.length > 1)
        self._half_bits = max(1, ((self.length - 1).bit_length() + 1) // 2)
        self._mask = (1 << self._half_bits) - 1
        generator = random.Random(int(seed))
        self._round_keys = tuple(generator.getrandbits(64) for _ in range(6))

    def _permute(self, value: int) -> int:
        left, right = value >> self._half_bits, value & self._mask
        for key in self._round_keys:
            mixed = (right ^ key) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
            mixed = (mixed ^ (mixed >> 30)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
            mixed ^= mixed >> 31
            left, right = right, left ^ (mixed & self._mask)
        return (left << self._half_bits) | right

    def __len__(self) -> int:
        """Return the logical index count."""
        return self.length

    def __getitem__(self, index: int | slice) -> int | list[int]:
        """Resolve one logical position or slice through the index plan."""
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError("Indexed source index plan out of range")
        if not self.shuffle:
            return index
        index = self._permute(index)
        while index >= self.length:
            index = self._permute(index)
        return index


class _RepeatedIndexPlan(Sequence[int]):
    """Repeat a source through a lazy deterministic permutation."""

    def __init__(self, source_length: int, target_length: int, seed: int, *, shuffle: bool) -> None:
        """Store a repeated index space without materializing it."""
        self.source_length = int(source_length)
        self.target_length = int(target_length)
        if self.source_length <= 0 or self.target_length < 0:
            raise ValueError("Repeated source length must be positive and target length non-negative")
        self.seed, self.shuffle = int(seed), shuffle
        self._epoch = 0
        self._plan = _IndexPermutation(self.source_length, self.seed, shuffle=shuffle)

    def __len__(self) -> int:
        """Return the logical index count."""
        return self.target_length

    def __getitem__(self, index: int | slice) -> int | list[int]:
        """Resolve one logical position or slice through the index plan."""
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self.target_length))]
        if index < 0:
            index += self.target_length
        if index < 0 or index >= self.target_length:
            raise IndexError("Indexed source repeated index plan out of range")
        epoch, source_index = divmod(index, self.source_length)
        if epoch != self._epoch:
            self._epoch = epoch
            self._plan = _IndexPermutation(self.source_length, self.seed + epoch, shuffle=self.shuffle)
        return int(self._plan[source_index])


def _balance_indices(
        source: _ReadCostProvider,
        indices: Sequence[int],
        *,
        metric: _ReadMetric,
        strategy: str,
        slots: int,
        group_size: int,
        layout: str,
) -> Sequence[int]:
    """Reorder indices to balance metadata costs across rank/worker slots.

    Each group is kept in its original position.  Only the ownership order
    inside that group changes, so a caller can preserve global micro-batch
    membership while assigning similarly sized reads to different DP ranks or
    workers.  The returned sequence is an index plan; no sample payload is
    opened by this function.

    ``layout='rank_major'`` emits one contiguous assignment per slot and is
    suitable for the existing Mapping batch sampler. ``layout='interleaved'``
    emits one item per slot in turn and is suitable for Iterable workers.
    """
    if strategy not in ("greedy", "lpt"):
        raise ValueError("read schedule strategy must be 'greedy' or 'lpt'")
    if slots <= 0:
        raise ValueError("read schedule slots must be positive")
    if group_size < slots or group_size % slots:
        raise ValueError("read schedule group_size must be a positive multiple of slots")
    if layout not in ("rank_major", "interleaved"):
        raise ValueError("read schedule layout must be rank_major or interleaved")

    _validate_cost_metric(metric)
    output = array("Q")
    for start in range(0, len(indices), group_size):
        block = list(indices[start:start + group_size])
        # A short rank-major tail cannot provide equal rank chunks. Preserve
        # it and let the existing sampler's drop_last policy decide.
        if layout == "rank_major" and len(block) < group_size:
            output.extend(block)
            continue
        assignments = _assign_balanced_block(source, block, metric, slots)
        rows = assignments if layout == "rank_major" else zip_longest(*assignments)
        output.extend(index for index in chain.from_iterable(rows) if index is not None)
    return output


def _validate_cost_metric(metric: _ReadMetric) -> None:
    """Reject unknown metadata fields and invalid weighted-cost coefficients."""
    supported = {"sample", "samples", "count", "bytes", "media_bytes", "pixels", "frames", "duration"}
    weights = metric if isinstance(metric, Mapping) else {metric: 1.0}
    for name, weight in weights.items():
        if str(name).lower() not in supported:
            raise ValueError(f"Unsupported read schedule metric {name!r}")
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Read schedule weight must be numeric: {name!r}") from error
        if not math.isfinite(numeric_weight) or numeric_weight < 0:
            raise ValueError(f"Read schedule weight must be finite and non-negative: {name!r}")


def _assign_balanced_block(
    source: _ReadCostProvider, block: Sequence[int], metric: _ReadMetric, slots: int,
) -> list[list[int]]:
    """Assign a bounded block by descending cost with fixed per-slot quotas."""
    costs = {index: source.cost_for_index(index, metric) for index in block}
    if any(not math.isfinite(cost) or cost < 0 for cost in costs.values()):
        raise ValueError("Read schedule costs must be finite and non-negative")
    quotient, remainder = divmod(len(block), slots)
    quotas = [quotient + (slot < remainder) for slot in range(slots)]
    assignments: list[list[int]] = [[] for _ in range(slots)]
    # Full slots leave the heap. Cost then slot ID preserve deterministic ties
    # without scanning every worker for every sample.
    available = [(0.0, slot) for slot, quota in enumerate(quotas) if quota]
    heapify(available)
    ranked = sorted(enumerate(block), key=lambda item: (-costs[item[1]], item[0]))
    for _, physical_index in ranked:
        load, slot = heappop(available)
        assignments[slot].append(physical_index)
        if len(assignments[slot]) < quotas[slot]:
            heappush(available, (load + costs[physical_index], slot))
    return assignments


class _BalancedIndexPlan(Sequence[int]):
    """Lazily balance an indexed source in bounded ownership windows.

    The plan intentionally keeps only the current window in memory.  A
    DataLoader worker can therefore resume or seek one output index without
    retaining a Python list of every sample in the epoch.
    """

    def __init__(
        self,
        source: IndexedSource,
        indices: Sequence[int],
        *,
        balance_by: str | Mapping[str, float] = "sample",
        balance_policy: str = "none",
        balance_slots: int = 1,
        balance_group_size: int,
        layout: str,
    ) -> None:
        """Cache one bounded metadata schedule window."""
        self.source = source
        self.indices = indices
        self.balance_by = balance_by
        self.balance_policy = balance_policy
        self.balance_slots = int(balance_slots)
        self.balance_group_size = int(balance_group_size)
        self.layout = layout
        self._cached_start: int | None = None
        self._cached_block: Sequence[int] | None = None
        if self.balance_slots <= 0:
            raise ValueError("Indexed source balance_slots must be positive")
        if (
            self.balance_group_size < self.balance_slots
            or self.balance_group_size % self.balance_slots
        ):
            raise ValueError(
                "Indexed source balance_group_size must be a positive multiple of balance_slots"
            )
        if not callable(getattr(source, "cost_for_index", None)):
            raise TypeError("Read scheduling requires a source with cost_for_index")
        _balance_indices(
            source, (), metric=balance_by, strategy=balance_policy, slots=self.balance_slots,
            group_size=self.balance_group_size, layout=layout,
        )

    def __len__(self) -> int:
        """Return the logical index count."""
        return len(self.indices)

    def _block_for(self, start: int) -> Sequence[int]:
        if self._cached_start == start and self._cached_block is not None:
            return self._cached_block
        block = list(self.indices[start : start + self.balance_group_size])
        balanced = _balance_indices(
            self.source,
            block,
            metric=self.balance_by,
            strategy=self.balance_policy,
            slots=self.balance_slots,
            group_size=self.balance_group_size,
            layout=self.layout,
        )
        self._cached_start = start
        self._cached_block = balanced
        return balanced

    def __getitem__(self, index: int | slice) -> int | list[int]:
        """Resolve one logical position or slice through the index plan."""
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Indexed source balanced plan out of range")
        start = (index // self.balance_group_size) * self.balance_group_size
        return int(self._block_for(start)[index - start])

    def __iter__(self) -> Iterable[int]:
        """Yield scheduled indices window by window without reading payloads."""
        for start in range(0, len(self), self.balance_group_size):
            yield from self._block_for(start)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cached_start"] = None
        state["_cached_block"] = None
        return state


class IndexedMappingView:
    """Expose a global deterministic index plan for the existing HP batch sampler."""

    def __init__(
        self,
        source: IndexedSource,
        *,
        size: int | None = None,
        seed: int = 42,
        shuffle: bool = False,
        balance_by: str | Mapping[str, float] = "sample",
        balance_policy: str = "none",
        balance_slots: int = 1,
        balance_group_size: int | None = None,
    ) -> None:
        """Plan global indices; leave rank ownership to the batch sampler."""
        self.source = source
        self.seed = int(seed)
        target_size = len(source) if size is None else int(size)
        if target_size < 0:
            raise ValueError("Indexed source size must be non-negative")
        if len(source) == 0 and target_size:
            raise ValueError("Indexed source split contains no samples")
        needs_shuffle = shuffle or target_size > len(source)
        if not needs_shuffle or not target_size:
            indices: Sequence[int] = range(target_size)
        else:
            indices = _RepeatedIndexPlan(
                len(source),
                target_size,
                self.seed,
                shuffle=needs_shuffle,
            )
        if balance_policy != "none":
            indices = _BalancedIndexPlan(
                source,
                indices,
                balance_by=balance_by,
                balance_policy=balance_policy,
                balance_slots=balance_slots,
                balance_group_size=balance_group_size or balance_slots,
                layout="rank_major",
            )
        self.indices = indices

    def validate_sampler(
        self, *, sampler_type: str, data_rearrange_map: Any = None,
        dp_world_size: int | None = None, micro_batch_size: int | None = None,
    ) -> None:
        """Validate the sampler ownership assumed by metadata balancing."""
        if isinstance(self.indices, _BalancedIndexPlan):
            if sampler_type != "single" or data_rearrange_map is not None:
                raise ValueError("Mapping read_balance requires sampler_type='single' and no data_rearrange_map")
            if dp_world_size is not None and self.indices.balance_slots != dp_world_size:
                raise ValueError("Mapping read_balance DP size does not match the batch sampler")
            if micro_batch_size is not None:
                expected_group_size = self.indices.balance_slots * micro_batch_size
                if self.indices.balance_group_size != expected_group_size:
                    raise ValueError("Mapping read_balance micro-batch size does not match the batch sampler")

    def __len__(self) -> int:
        """Return the logical index count."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        """Read the sample selected by one global logical index."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Indexed source Mapping view index out of range")
        return self.source[self.indices[index]]


class IndexedBlendMappingView:
    """Blend indexed sources with exact quotas and memory proportional to source count.

    A seeded global permutation scatters contiguous source quotas throughout
    the training plan. Source-local permutations select records without the
    full-length schedule and tiled shuffle arrays of an eager blend.
    """

    def __init__(
        self, source_datasets: Sequence[IndexedSource], weights: Sequence[float],
        size: int, random_seed: int = 42,
    ) -> None:
        """Store exact weighted quotas and lazy global/source-local permutations."""
        if not source_datasets or len(source_datasets) != len(weights):
            raise ValueError("Indexed blend sources and weights must be non-empty and aligned")
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("Indexed blend weights must be finite and positive")
        if any(len(source) <= 0 for source in source_datasets):
            raise ValueError("Indexed blend sources must not be empty")
        self.size = operator.index(size)
        if self.size < 0:
            raise ValueError("Indexed blend size must be non-negative")
        self.source_datasets = tuple(source_datasets)
        self._ends = []
        self._source_plans = []
        quotas = self._allocate_quotas(weights, self.size)
        cumulative = 0
        for source_id, (source, quota) in enumerate(zip(self.source_datasets, quotas)):
            cumulative += quota
            self._ends.append(cumulative)
            self._source_plans.append(_RepeatedIndexPlan(len(source), quota, random_seed + source_id, shuffle=True))
        self._plan = _IndexPermutation(self.size, random_seed, shuffle=True)

    @staticmethod
    def _allocate_quotas(weights: Sequence[float], size: int) -> list[int]:
        # Exact fractions avoid overflow and preserve quotas beyond float's
        # exact range; allocation runs once per source, not once per sample.
        exact_weights = [Fraction(str(weight)) for weight in weights]
        total = sum(exact_weights)
        shares = [weight * size / total for weight in exact_weights]
        quotas = [int(share) for share in shares]
        remainder_order = sorted(range(len(shares)), key=lambda index: (-(shares[index] - quotas[index]), index))
        for index in remainder_order[:size - sum(quotas)]:
            quotas[index] += 1
        return quotas

    def __len__(self) -> int:
        """Return the finite global training plan length."""
        return self.size

    def __getitem__(self, index: int) -> Any:
        """Resolve one global position without materializing blend schedules."""
        position = self._plan[index]
        source_id = bisect_right(self._ends, position)
        source_start = 0 if source_id == 0 else self._ends[source_id - 1]
        physical_index = self._source_plans[source_id][position - source_start]
        return self.source_datasets[source_id][physical_index]


class IndexedIterableView(IterableDataset):
    """Iterate disjoint DP/worker records and replay stable source indices.

    Finite epochs drop only the incomplete DP round. Workers split each rank's
    equal-length sequence, including when there are fewer records than workers.
    Cursor checkpoints require unchanged topology; physical replay keys do not.
    """

    def __init__(
        self,
        source: IndexedSource,
        *,
        context: Any = None,
        seed: int = 42,
        shuffle: bool = False,
        repeat: bool = False,
        output_index_for_resume: bool = False,
        balance_by: str | Mapping[str, float] = "sample",
        balance_policy: str = "none",
        balance_group_size: int | None = None,
        max_balance_group_size: int = _DEFAULT_MAX_BALANCE_GROUP_SIZE,
    ) -> None:
        """Configure deterministic iteration, ownership and replay."""
        self.source = source
        # Workers need DP coordinates, not the parent's process groups or callbacks.
        self.dp_rank = int(getattr(context, "dp_rank", 0))
        self.dp_world_size = int(getattr(context, "dp_world_size", 1))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.repeat = bool(repeat)
        self.output_index_for_resume = bool(output_index_for_resume)
        self.epoch = 0
        self._position = 0
        self._topology: tuple[int, int, int, int] | None = None
        self._shared_epoch = torch.zeros(1, dtype=torch.int64, device="cpu").share_memory_()
        self.balance_by = balance_by
        self.balance_policy = balance_policy
        self.balance_group_size = balance_group_size
        self.max_balance_group_size = int(max_balance_group_size)
        if self.max_balance_group_size <= 0:
            raise ValueError("Indexed source max_balance_group_size must be positive")
        if balance_group_size is not None and int(balance_group_size) <= 0:
            raise ValueError("Indexed source balance_group_size must be positive")
        if self.dp_world_size <= 0 or not 0 <= self.dp_rank < self.dp_world_size:
            raise ValueError("Indexed source requires positive dp_world_size and a valid dp_rank")
        if self.repeat and len(source) == 0:
            raise ValueError("Cannot repeat an empty indexed source")

    def set_epoch(self, epoch: int) -> None:
        """Publish a new epoch to persistent workers, preserving same-epoch resume.

        Call before creating the next DataLoader iterator, not while consuming
        an active iterator. Workers observe the shared value at iterator entry.
        """
        epoch = operator.index(epoch)
        if epoch < 0:
            raise ValueError("Indexed source epoch must be non-negative")
        self._shared_epoch[0] = epoch
        self._sync_epoch()

    def _sync_epoch(self) -> None:
        epoch = int(self._shared_epoch[0])
        if epoch != self.epoch:
            self.epoch = epoch
            self._position = 0
            self._topology = None

    def _worker_topology(self) -> tuple[int, int, int, int]:
        worker = torch_data.get_worker_info()
        return (
            self.dp_rank,
            self.dp_world_size,
            worker.id if worker is not None else 0,
            worker.num_workers if worker is not None else 1,
        )

    def __iter__(self) -> Any:
        """Yield rank-local records; replay indices identify physical records."""
        self._sync_epoch()
        topology = self._worker_topology()
        if self._topology is not None and topology != self._topology:
            raise ValueError("Indexed source cursor resume requires unchanged DP and worker topology")
        self._topology = topology
        dp_rank, dp_world, worker_id, worker_count = topology
        global_worker_id = worker_id * dp_world + dp_rank
        global_worker_count = dp_world * worker_count
        epoch_size = len(self.source)
        if not self.repeat:
            epoch_size -= epoch_size % dp_world
        if not epoch_size:
            return
        counter = self._position + (global_worker_id - self._position) % global_worker_count
        cycle = -1
        indices: Sequence[int] = ()
        while self.repeat or counter < epoch_size:
            next_cycle, plan_index = divmod(counter, epoch_size)
            if next_cycle != cycle:
                cycle = next_cycle
                indices = self._planned_indices(global_worker_count, cycle)
            source_index = indices[plan_index]
            counter += global_worker_count
            self._position = counter
            sample = self.source[source_index]
            if self.output_index_for_resume:
                yield sample, source_index
            else:
                yield sample

    def get_item(self, output_index: int) -> Any:
        """Replay one stable physical record independently of epoch or topology."""
        if isinstance(output_index, bool):
            raise TypeError("Indexed source replay index must be an integer")
        source_index = operator.index(output_index)
        if not 0 <= source_index < len(self.source):
            raise IndexError("Indexed source replay index out of range")
        return self.source[source_index]

    def _planned_indices(self, global_worker_count: int, cycle: int = 0) -> Sequence[int]:
        """Build one bounded plan, with a fresh shuffle for every repeated pass."""
        indices = _IndexPermutation(len(self.source), self.seed + self.epoch + cycle, shuffle=self.shuffle)
        if self.balance_policy == "none":
            return indices
        if self.balance_group_size is None:
            group_size = min(len(indices), self.max_balance_group_size)
            group_size = max(global_worker_count, group_size - group_size % global_worker_count)
        else:
            group_size = int(self.balance_group_size) * global_worker_count
        if group_size > self.max_balance_group_size:
            raise ValueError("Read schedule window exceeds max_balance_group_size; reduce workers or group size")
        return _BalancedIndexPlan(
            self.source,
            indices,
            balance_by=self.balance_by,
            balance_policy=self.balance_policy,
            balance_slots=global_worker_count,
            balance_group_size=group_size,
            layout="interleaved",
        )

    def _state_signature(self) -> dict[str, Any]:
        return {
            "length": len(self.source), "fingerprint": getattr(self.source, "fingerprint", None),
            "seed": self.seed, "shuffle": self.shuffle, "repeat": self.repeat,
            "balance_by": self.balance_by, "balance_policy": self.balance_policy,
            "balance_group_size": self.balance_group_size, "max_balance_group_size": self.max_balance_group_size,
        }

    def state_dict(self) -> dict[str, Any]:
        """Save a topology-bound cursor separately from replay identities."""
        self._sync_epoch()
        return {
            "version": 1, "position": self._position, "epoch": self.epoch,
            "topology": self._topology, "source": self._state_signature(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a matching source and defer worker-identity validation to iteration."""
        if state_dict.get("version") != 1 or state_dict.get("source") != self._state_signature():
            raise ValueError("Indexed source checkpoint does not match this source or iteration configuration")
        position, epoch = int(state_dict["position"]), int(state_dict["epoch"])
        if position < 0 or epoch < 0:
            raise ValueError("Indexed source checkpoint position and epoch must be non-negative")
        topology = state_dict.get("topology")
        if topology is not None:
            topology = tuple(topology)
            if len(topology) != 4:
                raise ValueError("Indexed source checkpoint has invalid worker topology")
        self._position, self.epoch = position, epoch
        self._topology = topology
        self._shared_epoch[0] = epoch


def _apply_sample_adapter(
    raw_sample: Any,
    sample_adapter: SampleAdapter | None,
    is_valid_sample: Callable[[Mapping[str, Any]], bool] | None = None,
) -> Mapping[str, Any]:
    """Apply an optional adapter and validate the canonical sample boundary."""
    adapted_sample = raw_sample if sample_adapter is None else sample_adapter(raw_sample)
    if not isinstance(adapted_sample, Mapping):
        raise TypeError(
            "SampleAdapter must return a Mapping canonical RawSample; "
            f"got {type(adapted_sample).__name__}"
        )
    if is_valid_sample is not None and not is_valid_sample(adapted_sample):
        raise ValueError(
            "Prepared source contains an invalid training sample; fix the dataset "
            "or explicitly enable data_config.filter_samples"
        )
    return adapted_sample


class _AdaptedMappingView:
    """Adapt raw indexed records, optionally materializing a filtered index."""

    def __init__(
        self, source: IndexedSource, adapter: SampleAdapter | None,
        is_valid_sample: Callable[[Mapping[str, Any]], bool] | None,
        *, filter_samples: bool = False,
    ) -> None:
        """Store the source and defer adapter decoding until indexing."""
        self.source = source
        self.adapter = adapter
        self.is_valid_sample = None if filter_samples else is_valid_sample
        # Only content filtering scans payloads; unfiltered adaptation stays lazy.
        self._filtered_indices = (
            self._build_filtered_indices(is_valid_sample) if filter_samples and is_valid_sample is not None else None
        )

    def _build_filtered_indices(self, is_valid_sample: Callable[[Mapping[str, Any]], bool]) -> array:
        indices = array("Q")
        for index, raw_sample in enumerate(self.source):
            sample = _apply_sample_adapter(raw_sample, self.adapter)
            if is_valid_sample(sample):
                indices.append(index)
        return indices

    def __len__(self) -> int:
        """Return the number of visible source samples."""
        return len(self.source) if self._filtered_indices is None else len(self._filtered_indices)

    def __getitem__(self, index: int) -> Any:
        """Decode and return one visible sample."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("adapted source Mapping view index out of range")
        source_index = index if self._filtered_indices is None else self._filtered_indices[index]
        return _apply_sample_adapter(self.source[source_index], self.adapter, self.is_valid_sample)


class _AdaptedIterableView(IterableDataset):
    """Adapt raw records while preserving the source cursor and replay identity.

    Repeating workers fail after one source length of consecutive rejections,
    bounding time spent in a filter that never produces a usable record.
    """

    def __init__(
        self, source: IndexedIterableView, adapter: SampleAdapter | None,
        is_valid_sample: Callable[[Mapping[str, Any]], bool] | None,
        *, filter_samples: bool = False,
    ) -> None:
        """Store the streaming source and sample conversion hooks."""
        self.source, self.adapter, self.is_valid_sample = source, adapter, is_valid_sample
        self.filter_samples = filter_samples

    @property
    def output_index_for_resume(self) -> bool:
        """Return whether the source emits replayable output indices."""
        return self.source.output_index_for_resume

    @output_index_for_resume.setter
    def output_index_for_resume(self, value: bool) -> None:
        """Enable or disable replayable output indices on the source."""
        self.source.output_index_for_resume = value

    def __iter__(self) -> Any:
        """Yield adapted records while retaining source output indices."""
        output_index_enabled = self.output_index_for_resume
        validator = None if self.filter_samples else self.is_valid_sample
        rejected = 0
        for item in self.source:
            output_index = None
            if output_index_enabled:
                item, output_index = item
            adapted = _apply_sample_adapter(item, self.adapter, validator)
            if self.filter_samples and self.is_valid_sample is not None and not self.is_valid_sample(adapted):
                rejected += 1
                if self.source.repeat and rejected >= len(self.source.source):
                    raise ValueError(
                        "Repeated indexed worker exceeded its consecutive rejected-sample limit; "
                        "prefilter sparse or invalid records in the prepared metadata"
                    )
                continue
            rejected = 0
            if output_index is None:
                yield adapted
            else:
                yield adapted, output_index

    def get_item(self, output_index: Any) -> Mapping[str, Any]:
        """Replay one RawSample for Online's existing transform wrapper."""
        adapted = _apply_sample_adapter(self.source.get_item(output_index), self.adapter)
        if self.is_valid_sample is not None and not self.is_valid_sample(adapted):
            raise ValueError("The replay index no longer identifies a visible source sample")
        return adapted

    def state_dict(self) -> dict[str, Any]:
        """Return the wrapped source checkpoint state."""
        return {"source": self.source.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the wrapped source checkpoint state."""
        self.source.load_state_dict(state_dict["source"])

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch changes to the wrapped source."""
        self.source.set_epoch(epoch)
