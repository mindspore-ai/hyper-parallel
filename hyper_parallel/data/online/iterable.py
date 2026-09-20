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
"""Build streaming Online sources and their local/global blend policy."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from torch.utils.data import IterableDataset  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import ONLINE_BLEND_STRATEGIES
from hyper_parallel.data.online.source import OnlineDataPath, _OnlineSourceBuilder
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext,
    build_dataset_for_dataloader,
    split_iterable_dataset_by_dp,
)

SampleTransform = Callable[[Any], list[Any]]


def build_online_iterable_source(
    *,
    data_config: Mapping[str, Any],
    data_path: OnlineDataPath | None = None,
    dataloader_context: DataLoaderParallelContext | None = None,
    sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> Any:
    """Build a locally shuffled and optionally DP-sharded Online stream.

    Args:
        data_config: Online streaming source options.
        data_path: Local path or Hub dataset ID. Use ``data_config.sources``
            instead for multiple sources.
        dataloader_context: Optional DataLoader ownership and DP policy.
        sample_filter: Optional raw-sample predicate applied before shuffling.

    Returns:
        One Iterable Dataset on each DataLoader-owning rank.
    """
    normalized_context = dataloader_context or DataLoaderParallelContext()
    if normalized_context.data_index_cache:
        normalized_context = replace(normalized_context, data_index_cache=False)

    dataset_builder = _OnlineIterableSourceBuilder(data_path, data_config, normalized_context)

    def dataset_factory() -> Any:
        """Build the configured single-source or multi-source stream."""
        iterable_dataset = dataset_builder.build_source(sample_filter=sample_filter)
        return iterable_dataset

    iterable_source = build_dataset_for_dataloader(
        dataset_factory,
        normalized_context,
        barrier_needed=False,
    )
    return iterable_source


def _supports_output_index_for_resume(dataset: Any) -> bool:
    """Return whether a Dataset can emit and rebuild stable output indices."""
    get_item = getattr(dataset, "get_item", None)
    supports_output_index = callable(get_item) and hasattr(dataset, "output_index_for_resume")
    return supports_output_index


class IterableTransformDataset(IterableDataset):
    """Apply a sample transform lazily while preserving streaming source state."""

    @classmethod
    def apply(cls, source_dataset: Any, transform: SampleTransform) -> Any:
        """Transform one Iterable Dataset built on a DataLoader-owning rank.

        Args:
            source_dataset: Iterable Dataset, or ``None`` on a non-owning rank.
            transform: Transform from one RawSample to zero, one, or many ModelSamples.

        Returns:
            A transformed Iterable Dataset, or ``None`` on a non-owning rank.
        """
        if source_dataset is None:
            return None

        iterable_dataset = cls(source_dataset, transform)
        return iterable_dataset

    def __init__(self, source_dataset: Any, transform: SampleTransform) -> None:
        """Store the streaming source and its lazy sample transform."""
        self.source_dataset = source_dataset
        self.transform = transform

    @property
    def output_index_for_resume(self) -> bool:
        """Return whether the upstream source emits stable output indices."""
        if not _supports_output_index_for_resume(self.source_dataset):
            raise AttributeError("The upstream iterable does not support output-index resume")

        output_index_enabled = bool(self.source_dataset.output_index_for_resume)
        return output_index_enabled

    @output_index_for_resume.setter
    def output_index_for_resume(self, value: bool) -> None:
        """Forward output-index mode to a replayable upstream source."""
        if not _supports_output_index_for_resume(self.source_dataset):
            raise ValueError("The upstream iterable does not support output-index resume")

        self.source_dataset.output_index_for_resume = value

    def get_item(self, output_index: Any) -> list[Any]:
        """Rebuild and transform one upstream item from its stable output index."""
        if not _supports_output_index_for_resume(self.source_dataset):
            raise AttributeError("The upstream iterable does not support get_item")

        raw_sample = self.source_dataset.get_item(output_index)
        transformed_samples = self._transform_sample(raw_sample)
        return transformed_samples

    def __iter__(self) -> Any:
        """Transform source records lazily while preserving source order."""
        output_index_enabled = (
            _supports_output_index_for_resume(self.source_dataset)
            and bool(self.source_dataset.output_index_for_resume)
        )
        for source_item in self.source_dataset:
            if output_index_enabled:
                raw_sample, output_index = source_item
            else:
                raw_sample = source_item

            transformed_samples = self._transform_sample(raw_sample)
            if not transformed_samples:
                continue
            if output_index_enabled:
                yield transformed_samples, output_index
            elif len(transformed_samples) == 1:
                yield transformed_samples[0]
            else:
                yield transformed_samples

    def state_dict(self) -> dict[str, Any]:
        """Forward checkpoint state to a stateful upstream stream."""
        state_builder = getattr(self.source_dataset, "state_dict", None)
        source_state = {}
        if callable(state_builder):
            source_state = state_builder()
        transform_state = {"source_dataset": source_state}
        return transform_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore checkpoint state through the upstream stream interface."""
        state_loader = getattr(self.source_dataset, "load_state_dict", None)
        if not callable(state_loader):
            raise ValueError("Online iterable source does not support load_state_dict")

        state_loader(state_dict["source_dataset"])

    def set_epoch(self, epoch: int) -> None:
        """Forward deterministic epoch state when supported upstream."""
        epoch_setter = getattr(self.source_dataset, "set_epoch", None)
        if callable(epoch_setter):
            epoch_setter(epoch)

    def _transform_sample(self, raw_sample: Any) -> list[Any]:
        """Transform one RawSample and enforce the zero/one/many contract."""
        transformed_samples = self.transform(raw_sample)
        if not isinstance(transformed_samples, list):
            raise TypeError("Online Iterable sample transform must return a list")

        return transformed_samples


class _LocalWeightedIterableDataset(IterableDataset):
    """Choose among already-sharded Iterable sources on the local rank."""

    def __init__(
        self,
        source_datasets: Sequence[Any],
        weights: Sequence[float],
        *,
        random_seed: int,
        stopping_strategy: str,
        vary_by_epoch: bool,
    ) -> None:
        """Store the rank-local sources and weighted selection policy."""
        self.source_datasets = tuple(source_datasets)
        self.weights = tuple(weights)
        self.random_seed = random_seed
        self.stopping_strategy = stopping_strategy
        self.vary_by_epoch = vary_by_epoch
        self.epoch = 0
        self._random_state = None
        self._active_indices = None
        self._exhausted_once = None

    def __iter__(self) -> Any:
        """Yield samples according to the configured local source weights."""
        source_epoch = getattr(self.source_datasets[0], "epoch", self.epoch)
        epoch = 0
        if self.vary_by_epoch:
            epoch = int(source_epoch)

        random_generator = random.Random(self.random_seed + epoch)
        if self._random_state is not None:
            random_generator.setstate(self._random_state)

        source_iterators = []
        for source_dataset in self.source_datasets:
            source_iterators.append(iter(source_dataset))

        if self._active_indices is None:
            active_indices = list(range(len(source_iterators)))
        else:
            active_indices = list(self._active_indices)

        if self._exhausted_once is None:
            exhausted_once = set()
        else:
            exhausted_once = set(self._exhausted_once)

        while active_indices:
            active_weights = []
            for active_index in active_indices:
                active_weights.append(self.weights[active_index])

            source_index = random_generator.choices(active_indices, weights=active_weights, k=1)[0]
            try:
                source_sample = next(source_iterators[source_index])
            except StopIteration:
                if self.stopping_strategy == "first_exhausted":
                    self._clear_iteration_state()
                    return

                exhausted_once.add(source_index)
                if self.stopping_strategy == "all_exhausted_without_replacement":
                    active_indices.remove(source_index)
                    self._save_iteration_state(random_generator, active_indices, exhausted_once)
                    continue

                if len(exhausted_once) == len(source_iterators):
                    self._clear_iteration_state()
                    return

                source_iterators[source_index] = iter(self.source_datasets[source_index])
                self._save_iteration_state(random_generator, active_indices, exhausted_once)
                continue

            self._save_iteration_state(random_generator, active_indices, exhausted_once)
            yield source_sample

        self._clear_iteration_state()

    def set_epoch(self, epoch: int) -> None:
        """Set the local blend epoch and forward it to every source."""
        self.epoch = epoch
        self._clear_iteration_state()
        for source_dataset in self.source_datasets:
            epoch_setter = getattr(source_dataset, "set_epoch", None)
            if callable(epoch_setter):
                epoch_setter(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Save source cursors and the local weighted-selection cursor."""
        source_states = []
        for source_dataset in self.source_datasets:
            state_builder = getattr(source_dataset, "state_dict", None)
            source_state = None
            if callable(state_builder):
                source_state = state_builder()
            source_states.append(source_state)

        source_epoch = getattr(self.source_datasets[0], "epoch", self.epoch)
        dataset_state = {
            "epoch": int(source_epoch),
            "random_state": self._random_state,
            "active_indices": self._active_indices,
            "exhausted_once": self._exhausted_once,
            "sources": source_states,
        }
        return dataset_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore source cursors and the local weighted-selection cursor."""
        source_states = state_dict["sources"]
        if len(source_states) != len(self.source_datasets):
            raise ValueError("Online source count does not match the saved local blend state")

        for source_dataset, source_state in zip(self.source_datasets, source_states):
            if source_state is None:
                continue
            state_loader = getattr(source_dataset, "load_state_dict", None)
            if not callable(state_loader):
                raise ValueError("Online iterable source does not support load_state_dict")
            state_loader(source_state)

        self.epoch = int(state_dict["epoch"])
        self._random_state = state_dict["random_state"]
        self._active_indices = state_dict["active_indices"]
        self._exhausted_once = state_dict["exhausted_once"]

    def _save_iteration_state(
        self,
        random_generator: random.Random,
        active_indices: Sequence[int],
        exhausted_once: set[int],
    ) -> None:
        """Record the local source-selection cursor after one decision."""
        self._random_state = random_generator.getstate()
        self._active_indices = tuple(active_indices)
        self._exhausted_once = tuple(sorted(exhausted_once))

    def _clear_iteration_state(self) -> None:
        """Reset the local source-selection cursor for a new epoch."""
        self._random_state = None
        self._active_indices = None
        self._exhausted_once = None


class _OnlineIterableSourceBuilder(_OnlineSourceBuilder):
    """Build the rank-local or globally interleaved streaming source graph."""

    def __init__(
        self,
        data_path: OnlineDataPath | None,
        data_config: Mapping[str, Any],
        dataloader_context: DataLoaderParallelContext,
    ) -> None:
        """Store source configuration and DataLoader ownership."""
        super().__init__(data_path, data_config)
        self.dataloader_context = dataloader_context

    def build_source(
        self,
        *,
        sample_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Any:
        """Load Iterable sources and apply their shard, shuffle, and blend policy."""
        source_loaders, weights = self._create_source_loaders()
        source_datasets = []
        for source_loader in source_loaders:
            source_dataset = source_loader.load(streaming=True, sample_filter=sample_filter)
            if isinstance(source_dataset, tuple):
                raise ValueError("Online Iterable Dataset requires one train source per entry")
            source_datasets.append(source_dataset)

        if len(source_datasets) == 1:
            iterable_dataset = self._prepare_local_source(source_datasets[0])
            return iterable_dataset

        blend_strategy = str(self.data_config.get("blend_strategy", "local_weighted"))
        if blend_strategy not in ONLINE_BLEND_STRATEGIES:
            raise ValueError(
                f"Unsupported Online blend_strategy {blend_strategy!r}; "
                f"expected one of {ONLINE_BLEND_STRATEGIES!r}"
            )

        if blend_strategy == "local_weighted":
            local_sources = []
            for source_dataset in source_datasets:
                local_source = self._prepare_local_source(source_dataset)
                local_sources.append(local_source)

            iterable_dataset = _LocalWeightedIterableDataset(
                local_sources,
                weights,
                random_seed=self._random_seed,
                stopping_strategy=self._stopping_strategy,
                vary_by_epoch=self._shuffle_enabled,
            )
            return iterable_dataset

        shuffled_sources = []
        for source_dataset in source_datasets:
            shuffled_source = self._shuffle(source_dataset)
            shuffled_sources.append(shuffled_source)

        iterable_dataset = self._interleave_sources(shuffled_sources, weights)
        if self._shuffle_enabled:
            # buffer_size=1 preserves output order while making set_epoch reseed the full source graph.
            iterable_dataset = iterable_dataset.shuffle(seed=self._random_seed, buffer_size=1)
        if self._split_by_data_parallel:
            iterable_dataset = split_iterable_dataset_by_dp(iterable_dataset, self.dataloader_context)
        return iterable_dataset

    @property
    def _shuffle_enabled(self) -> bool:
        """Return whether Iterable sources use local buffer shuffle."""
        return bool(self.data_config.get("shuffle", True))

    @property
    def _split_by_data_parallel(self) -> bool:
        """Return whether Iterable sources own DP sharding."""
        return bool(self.data_config.get("split_by_data_parallel", True))

    def _prepare_local_source(self, source_dataset: Any) -> Any:
        """Apply per-source DP sharding before local buffer shuffle."""
        if self._split_by_data_parallel:
            source_dataset = split_iterable_dataset_by_dp(source_dataset, self.dataloader_context)
        local_dataset = self._shuffle(source_dataset)
        return local_dataset

    def _shuffle(self, source_dataset: Any) -> Any:
        """Apply deterministic buffer shuffle when configured."""
        if not self._shuffle_enabled:
            return source_dataset

        buffer_size = int(self.data_config.get("shuffle_buffer_size", 10_000))
        if buffer_size <= 0:
            raise ValueError("Online shuffle_buffer_size must be positive")

        shuffled_dataset = source_dataset.shuffle(seed=self._random_seed, buffer_size=buffer_size)
        return shuffled_dataset


__all__ = [
    "IterableTransformDataset",
    "build_online_iterable_source",
]
