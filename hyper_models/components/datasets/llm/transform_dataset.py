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

from torch.utils.data import IterableDataset

from hyper_models.components.datasets.contracts import (
    ModelSample,
    SampleTransform,
    is_iterable_dataset,
)
from hyper_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_models.components.utils.constants import IGNORE_INDEX

logger = get_dataset_logger(__name__)


def _has_trainable_labels(model_sample: Mapping[str, Any]) -> bool:
    """Return whether causal shifting leaves at least one trainable target."""
    labels = model_sample.get("labels")
    if labels is None:
        return True
    shifted_labels = labels[1:]
    if hasattr(labels, "ne"):
        return bool(shifted_labels.ne(IGNORE_INDEX).any())
    return any(label != IGNORE_INDEX for label in shifted_labels)


def _normalize_transformed_samples(transformed_sample: Any) -> list[ModelSample]:
    """Normalize a transform result without recomputing it."""
    if isinstance(transformed_sample, Mapping):
        return [dict(transformed_sample)]

    if isinstance(transformed_sample, Sequence) and not isinstance(transformed_sample, (str, bytes)):
        model_samples = []
        for model_sample in transformed_sample:
            if not isinstance(model_sample, Mapping):
                raise ValueError("Every transformed LLM sample must be a mapping")
            model_samples.append(dict(model_sample))
        return model_samples

    raise ValueError("An LLM transform must return a mapping or a sequence of mappings")


def _transform_sample(raw_sample: Any, transform: SampleTransform | None) -> ModelSample:
    """Apply the transform and require exactly one model sample per record.

    Args:
        raw_sample: Source record read from the underlying Dataset.
        transform: Optional Trainer-built sample transform.

    Returns:
        The single normalized model sample.

    Raises:
        ValueError: If the transform produces zero or multiple model samples.
    """
    transformed_sample = transform(raw_sample) if transform is not None else raw_sample
    model_samples = _normalize_transformed_samples(transformed_sample)
    if len(model_samples) != 1:
        raise ValueError(
            "An LLM transform must currently produce exactly one model sample per source record; "
            "multi-sample expansion requires the deferred packing Dataset stage"
        )
    return model_samples[0]


class _LLMTransformDataset:
    """Apply one Trainer-built transform after source-specific IO."""

    def __init__(
            self,
            source_dataset: Any,
            transform: SampleTransform | None,
            *,
            skip_invalid_samples: bool = False,
    ) -> None:
        """Store the source Dataset and its LLM sample transform."""
        self.source_dataset = source_dataset
        self.transform = transform
        self.skip_invalid_samples = skip_invalid_samples

    def __len__(self) -> int:
        """Return the source Dataset length."""
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> ModelSample:
        """Read one RawSample and normalize one transformed ModelSample."""
        if not self.skip_invalid_samples:
            raw_sample = self.source_dataset[index]
            return _transform_sample(raw_sample, self.transform)

        dataset_length = len(self.source_dataset)
        if index < 0:
            index += dataset_length
        if index < 0 or index >= dataset_length:
            raise IndexError("LLM Dataset index out of range")

        for offset in range(dataset_length):
            source_index = (index + offset) % dataset_length
            raw_sample = self.source_dataset[source_index]
            transformed_sample = self.transform(raw_sample) if self.transform is not None else raw_sample
            model_samples = _normalize_transformed_samples(transformed_sample)
            trainable_samples = [sample for sample in model_samples if _has_trainable_labels(sample)]
            if len(trainable_samples) == 1:
                return trainable_samples[0]
            if len(trainable_samples) > 1:
                raise ValueError(
                    "An LLM transform must currently produce exactly one trainable model sample per source record; "
                    "multi-sample expansion requires the deferred packing Dataset stage"
                )
        raise ValueError("LLM Dataset contains no samples with trainable labels")


class _LLMIterableTransformDataset(IterableDataset):
    """Reserve the common transform boundary for an online streaming source."""

    def __init__(
            self,
            source_dataset: Any,
            transform: SampleTransform | None,
            *,
            skip_invalid_samples: bool = False,
    ) -> None:
        """Store the streaming source and its LLM sample transform."""
        self.source_dataset = source_dataset
        self.transform = transform
        self.skip_invalid_samples = skip_invalid_samples

    def __iter__(self) -> Any:
        """Transform RawSamples lazily while preserving upstream order."""
        for raw_sample in self.source_dataset:
            model_sample = _transform_sample(raw_sample, self.transform)
            if self.skip_invalid_samples and not _has_trainable_labels(model_sample):
                continue
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


def _wrap_llm_dataset(
        dataset: Any,
        transform: SampleTransform | None,
        *,
        skip_invalid_samples: bool = False,
) -> Any:
    """Wrap one source Dataset with the appropriate LLM transform wrapper.

    Args:
        dataset: Map-style or iterable source Dataset.
        transform: Optional Trainer-built sample transform.
        skip_invalid_samples: Whether to skip records without trainable labels.

    Returns:
        The wrapped Dataset applying ``transform`` to each source record.

    Raises:
        ValueError: If the source Dataset is neither map-style nor iterable.
    """
    if (is_iterable_dataset(dataset) and callable(getattr(dataset, "map", None)) and not skip_invalid_samples):
        transform_callable = _OnlineIterableTransform(transform)
        column_names = getattr(dataset, "column_names", None)
        map_options = {}
        if column_names:
            map_options["remove_columns"] = column_names
        transformed_dataset = dataset.map(transform_callable, **map_options)
        logger.debug("Wrapped map-capable iterable Dataset with transform=%s", type(transform).__name__)
        return transformed_dataset

    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        transformed_dataset = _LLMTransformDataset(
            dataset,
            transform,
            skip_invalid_samples=skip_invalid_samples,
        )
        logger.debug("Wrapped mapping Dataset with transform=%s", type(transform).__name__)
        return transformed_dataset

    if hasattr(dataset, "__iter__"):
        transformed_dataset = _LLMIterableTransformDataset(
            dataset,
            transform,
            skip_invalid_samples=skip_invalid_samples,
        )
        logger.debug("Wrapped iterable Dataset with transform=%s", type(transform).__name__)
        return transformed_dataset

    raise ValueError("LLM source Dataset must be map-style or iterable")


def apply_llm_data_transform(
        dataset_result: Any,
        transform: SampleTransform | None,
        *,
        skip_invalid_samples: bool = False,
) -> Any:
    """Apply the common transform stage to one Dataset or three indexed splits.

    Args:
        dataset_result: Online RawSample Dataset or offline indexed splits.
        transform: Plaintext/conversation transform for Online, or
            ``PretokenizedTransform`` for Offline.
        skip_invalid_samples: Whether to skip records whose transformed sample
            has no trainable label after causal shifting.

    Returns:
        The same Dataset result shape with each available Dataset wrapped.
    """
    if isinstance(dataset_result, tuple) and len(dataset_result) == 3:
        transformed_splits = tuple(
            None if dataset is None else _wrap_llm_dataset(
                dataset,
                transform,
                skip_invalid_samples=skip_invalid_samples,
            )
            for dataset in dataset_result
        )
        return transformed_splits

    if dataset_result is None:
        return None

    transformed_dataset = _wrap_llm_dataset(
        dataset_result,
        transform,
        skip_invalid_samples=skip_invalid_samples,
    )
    return transformed_dataset
