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
"""Lazy plaintext or conversation transforms for Online LLM sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from torch.utils.data import IterableDataset  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.dataset_logging import (
    get_dataset_logger,
)
from hyper_parallel.data.constants import IGNORE_INDEX

logger = get_dataset_logger(__name__)
ModelSample: TypeAlias = Mapping[str, Any]
SampleTransform: TypeAlias = Callable[[Any], Any]


def _supports_output_index_for_resume(dataset: Any) -> bool:
    """Return whether a Dataset can emit and rebuild stable output indices."""
    get_item = getattr(dataset, "get_item", None)
    return callable(get_item) and hasattr(dataset, "output_index_for_resume")


def _has_trainable_labels(model_sample: Mapping[str, Any]) -> bool:
    """Return whether pre-shifted labels contain at least one trainable target."""
    labels = model_sample.get("labels")
    if labels is None:
        return True
    if hasattr(labels, "ne"):
        return bool(labels.ne(IGNORE_INDEX).any())
    return any(label != IGNORE_INDEX for label in labels)


def _normalize_transformed_samples(transformed_sample: Any) -> list[ModelSample]:
    """Normalize a transform result without recomputing it."""
    if isinstance(transformed_sample, Mapping):
        model_samples = [dict(transformed_sample)]
        return model_samples

    if isinstance(transformed_sample, Sequence) and not isinstance(transformed_sample, (str, bytes)):
        model_samples = []
        for model_sample in transformed_sample:
            if not isinstance(model_sample, Mapping):
                raise ValueError("Every transformed LLM sample must be a mapping")

            model_samples.append(dict(model_sample))
        return model_samples

    raise ValueError("An LLM transform must return a mapping or a sequence of mappings")


def _transform_samples(raw_sample: Any, transform: SampleTransform | None) -> list[ModelSample]:
    """Transform one source item into its ordered ModelSamples."""
    transformed_sample = transform(raw_sample) if transform is not None else raw_sample
    model_samples = _normalize_transformed_samples(transformed_sample)
    return model_samples


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
        dataset_length = len(self.source_dataset)
        return dataset_length

    def get_item(self, index: int) -> list[ModelSample]:
        """Transform one source index into its ordered trainable ModelSamples.

        Args:
            index: Mapping source index to transform exactly once.

        Returns:
            Ordered trainable ModelSamples, possibly empty for an invalid item.
        """
        dataset_length = len(self.source_dataset)
        if index < 0:
            index += dataset_length
        if index < 0 or index >= dataset_length:
            raise IndexError("LLM Dataset index out of range")

        raw_sample = self.source_dataset[index]
        model_samples = _transform_samples(raw_sample, self.transform)
        if self.skip_invalid_samples:
            model_samples = [sample for sample in model_samples if _has_trainable_labels(sample)]

        return model_samples

    def __getitem__(self, index: int) -> ModelSample:
        """Return one ModelSample for fixed-size DataLoader consumers."""
        model_samples = self.get_item(index)
        if not model_samples:
            raise ValueError(
                "An invalid mapping source item requires source filtering or DynamicBatchDataLoader"
            )

        if len(model_samples) != 1:
            raise ValueError(
                "A multi-sample source item requires DynamicBatchDataLoader"
            )

        model_sample = model_samples[0]
        return model_sample


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

    @property
    def output_index_for_resume(self) -> bool:
        """Return whether the upstream source emits an output index with each item."""
        if not _supports_output_index_for_resume(self.source_dataset):
            raise AttributeError("The upstream iterable does not support output-index resume")

        return bool(self.source_dataset.output_index_for_resume)

    @output_index_for_resume.setter
    def output_index_for_resume(self, value: bool) -> None:
        """Forward output-index mode to a replayable upstream source.

        Args:
            value: Whether the upstream source should emit stable output indices.
        """
        if not _supports_output_index_for_resume(self.source_dataset):
            raise ValueError("The upstream iterable does not support output-index resume")

        self.source_dataset.output_index_for_resume = value

    def get_item(self, output_index: Any) -> list[ModelSample]:
        """Rebuild and transform one upstream item from its stable output index.

        Args:
            output_index: Stable replay key emitted by the upstream source.

        Returns:
            Ordered trainable ModelSamples rebuilt from the source item.
        """
        if not _supports_output_index_for_resume(self.source_dataset):
            raise AttributeError("The upstream iterable does not support get_item")

        raw_sample = self.source_dataset.get_item(output_index)
        model_samples = self._prepare_model_samples(raw_sample)
        return model_samples

    def _prepare_model_samples(self, raw_sample: Any) -> list[ModelSample]:
        """Transform and filter one upstream item without changing its order."""
        model_samples = _transform_samples(raw_sample, self.transform)
        if self.skip_invalid_samples:
            model_samples = [sample for sample in model_samples if _has_trainable_labels(sample)]
        return model_samples

    def __iter__(self) -> Any:
        """Transform RawSamples lazily while preserving upstream order."""
        output_index_enabled = (
            _supports_output_index_for_resume(self.source_dataset)
            and bool(self.source_dataset.output_index_for_resume)
        )
        for source_item in self.source_dataset:
            if output_index_enabled:
                raw_sample, output_index = source_item
            else:
                raw_sample = source_item

            model_samples = self._prepare_model_samples(raw_sample)
            if not model_samples:
                continue
            if output_index_enabled:
                yield model_samples, output_index
            elif len(model_samples) == 1:
                yield model_samples[0]
            else:
                yield model_samples

    def state_dict(self) -> dict[str, Any]:
        """Forward checkpoint state to a stateful upstream stream."""
        state_builder = getattr(self.source_dataset, "state_dict", None)
        source_state = state_builder() if callable(state_builder) else {}
        state = {"source_dataset": source_state}
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore checkpoint state through the upstream stream interface.

        Args:
            state_dict: Checkpoint state containing the upstream Dataset state.
        """
        state_loader = getattr(self.source_dataset, "load_state_dict", None)
        if not callable(state_loader):
            raise ValueError("Online iterable source does not support load_state_dict")

        state_loader(state_dict["source_dataset"])

    def set_epoch(self, epoch: int) -> None:
        """Forward deterministic epoch state when supported upstream.

        Args:
            epoch: Epoch number forwarded to the upstream source.
        """
        epoch_setter = getattr(self.source_dataset, "set_epoch", None)
        if callable(epoch_setter):
            epoch_setter(epoch)


def _wrap_llm_dataset(
        dataset: Any,
        transform: SampleTransform | None,
        *,
        skip_invalid_samples: bool = False,
) -> Any:
    """Wrap a mapping or iterable Dataset with the configured sample transform."""
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
    """Apply the common transform stage to an Online Dataset result.

    Args:
        dataset_result: Online RawSample Dataset or split tuple.
        transform: Plaintext or conversation Online transform.
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
