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
"""Common transform Dataset for online and offline LLM sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyper_models.components.datasets.contracts import (
    ModelSample,
    SampleTransform,
    is_iterable_dataset,
)


def _transform_sample(raw_sample: Any, transform: SampleTransform | None) -> ModelSample:
    transformed_sample = transform(raw_sample) if transform is not None else raw_sample
    if isinstance(transformed_sample, Mapping):
        model_sample = dict(transformed_sample)
        return model_sample
    if isinstance(transformed_sample, Sequence) and not isinstance(transformed_sample, (str, bytes)):
        if len(transformed_sample) != 1:
            raise ValueError(
                "An LLM transform must currently produce exactly one model sample per source record; "
                "multi-sample expansion requires the deferred packing Dataset stage"
            )
        model_sample = transformed_sample[0]
        if isinstance(model_sample, Mapping):
            normalized_sample = dict(model_sample)
            return normalized_sample
    raise ValueError("An LLM transform must return a mapping or a single-item sequence of mappings")


class _LLMTransformDataset:
    """Apply one Trainer-built transform after source-specific IO."""

    def __init__(self, source_dataset: Any, transform: SampleTransform | None) -> None:
        """Store the source Dataset and its LLM sample transform."""
        self.source_dataset = source_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the source Dataset length."""
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> ModelSample:
        """Read one RawSample and normalize one transformed ModelSample."""
        raw_sample = self.source_dataset[index]
        model_sample = _transform_sample(raw_sample, self.transform)
        return model_sample


class _LLMIterableTransformDataset:
    """Reserve the common transform boundary for an online streaming source."""

    def __init__(self, source_dataset: Any, transform: SampleTransform | None) -> None:
        """Store the streaming source and its LLM sample transform."""
        self.source_dataset = source_dataset
        self.transform = transform

    def __iter__(self) -> Any:
        """Transform RawSamples lazily while preserving upstream order."""
        for raw_sample in self.source_dataset:
            model_sample = _transform_sample(raw_sample, self.transform)
            yield model_sample

    def state_dict(self) -> dict[str, Any]:
        """Forward checkpoint state to a stateful upstream stream."""
        state_builder = getattr(self.source_dataset, "state_dict", None)
        source_state = state_builder() if callable(state_builder) else {}
        state = {"source_dataset": source_state}
        return state

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


class _OnlineIterableTransform:
    """Pickleable Hugging Face map callable for one Online RawSample."""

    def __init__(self, transform: SampleTransform | None) -> None:
        """Store the common LLM transform."""
        self.transform = transform

    def __call__(self, raw_sample: Any) -> ModelSample:
        """Convert one streaming RawSample into one ModelSample."""
        model_sample = _transform_sample(raw_sample, self.transform)
        return model_sample


def _wrap_llm_dataset(dataset: Any, transform: SampleTransform | None) -> Any:
    if is_iterable_dataset(dataset) and callable(getattr(dataset, "map", None)):
        transform_callable = _OnlineIterableTransform(transform)
        column_names = getattr(dataset, "column_names", None)
        map_options = {}
        if column_names:
            map_options["remove_columns"] = column_names
        transformed_dataset = dataset.map(transform_callable, **map_options)
        return transformed_dataset

    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        transformed_dataset = _LLMTransformDataset(dataset, transform)
        return transformed_dataset

    if hasattr(dataset, "__iter__"):
        transformed_dataset = _LLMIterableTransformDataset(dataset, transform)
        return transformed_dataset

    raise ValueError("LLM source Dataset must be map-style or iterable")


def apply_llm_data_transform(dataset_result: Any, transform: SampleTransform | None) -> Any:
    """Apply the common transform stage to one Dataset or three indexed splits.

    Args:
        dataset_result: Online RawSample Dataset or offline indexed splits.
        transform: Plaintext/conversation transform for Online, or
            ``PretokenizedTransform`` for Offline.

    Returns:
        The same Dataset result shape with each available Dataset wrapped.
    """
    if isinstance(dataset_result, tuple) and len(dataset_result) == 3:
        transformed_splits = tuple(
            None if dataset is None else _wrap_llm_dataset(dataset, transform)
            for dataset in dataset_result
        )
        return transformed_splits
    if dataset_result is None:
        return None
    transformed_dataset = _wrap_llm_dataset(dataset_result, transform)
    return transformed_dataset
