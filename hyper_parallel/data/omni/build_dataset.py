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
"""Compose an Online source with the configured Omni transform lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.constants import ONLINE_SPLIT_COUNT
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.omni.omni_transform import (
    OmniDataTransform,
    _OmniTransformStrategy,
)
from hyper_parallel.data.online import (
    MappingTransformDataset,
    OnlineDataPath,
    build_online_mapping_source,
)

logger = get_dataset_logger(__name__)


def _get_train_valid_test_num_samples(training_config: Any) -> tuple[int, int, int]:
    """Calculate train, validation, and test sizes from the training plan."""
    if training_config is None:
        raise ValueError("Dataset sample planning requires a training configuration")

    global_batch_size = training_config.global_batch_size
    if training_config.train_iters is not None:
        train_iters = training_config.train_iters
    elif training_config.train_samples:
        train_iters = training_config.train_samples // global_batch_size
    else:
        raise ValueError("training.train_iters and training.train_samples cannot both be None")

    train_samples = training_config.train_samples or train_iters * global_batch_size
    eval_iters = training_config.eval_iters
    valid_iters = (train_iters // eval_iters + 1) * eval_iters if eval_iters else 0
    split_sizes = (train_samples, valid_iters * global_batch_size, eval_iters * global_batch_size)
    logger.debug("Dataset target sizes: train=%d, validation=%d, test=%d", *split_sizes)
    return split_sizes


def build_online_omni_mapping_dataset(
    *,
    data_config: Mapping[str, Any],
    data_path: OnlineDataPath | None = None,
    transform: OmniDataTransform | None = None,
    training_config: Any = None,
) -> Any:
    """Build an Online Mapping source and apply its Omni transform lazily.

    Args:
        data_config: Online Mapping source options.
        data_path: Optional local source path, ordered paths, or pre-split
            train/valid/test path mapping.
        transform: Omni sample transform selected by the Trainer.
        training_config: Training plan providing the random seed and split sizes.

    Returns:
        A transformed Online Mapping Dataset or train-valid-test tuple.

    Raises:
        TypeError: If transform is not an OmniDataTransform.
        ValueError: If no Omni transform is configured.
    """
    if transform is None:
        raise ValueError("Online Omni Dataset requires a data_transform")
    if not isinstance(transform, OmniDataTransform):
        raise TypeError("Online Omni Dataset transform must be an OmniDataTransform")

    dataset_config = dict(data_config)
    training_seed = getattr(training_config, "seed", None)
    dataset_config["random_seed"] = 42 if training_seed is None else int(training_seed)
    train_valid_test_num_samples = None
    if training_config is not None:
        train_valid_test_num_samples = _get_train_valid_test_num_samples(training_config)

    sample_filter = transform.is_valid_sample
    source_dataset = build_online_mapping_source(
        data_path=data_path,
        data_config=dataset_config,
        sample_filter=sample_filter,
        train_valid_test_num_samples=train_valid_test_num_samples,
    )
    transformed_dataset = _OmniMappingDataset.apply(source_dataset, transform)
    return transformed_dataset


class _OmniMappingDataset(MappingTransformDataset):
    """Retain the Omni encoding lifecycle around one Mapping source."""

    def __init__(self, source_dataset: Any, transform: OmniDataTransform) -> None:
        """Build the pre-selection strategy and retain post-selection hooks."""
        self.data_transform = transform
        self._transform_strategy = _OmniTransformStrategy.from_transform(transform)
        super().__init__(source_dataset, self._transform_strategy.prepare_samples)

    @classmethod
    def apply(cls, source_dataset: Any, transform: OmniDataTransform) -> Any:
        """Wrap one Mapping source or each available train-valid-test split."""
        if isinstance(source_dataset, tuple) and len(source_dataset) == ONLINE_SPLIT_COUNT:
            transformed_splits = []
            for split_dataset in source_dataset:
                transformed_dataset = None
                if split_dataset is not None:
                    transformed_dataset = cls(split_dataset, transform)
                transformed_splits.append(transformed_dataset)

            mapping_splits = (
                transformed_splits[0],
                transformed_splits[1],
                transformed_splits[2],
            )
            return mapping_splits

        if source_dataset is None:
            return None

        mapping_dataset = cls(source_dataset, transform)
        return mapping_dataset

    def encode_selected_sample(self, sample: Any) -> Any:
        """Finish encoding one sample after packing selection."""
        encoded_sample = self._transform_strategy.encode_selected_sample(sample)
        return encoded_sample

    def encode_batch(self, batch: Any) -> Any:
        """Apply the transform's final batch encoding hook."""
        encoded_batch = self.data_transform.encode_batch(batch)
        return encoded_batch


__all__ = ["build_online_omni_mapping_dataset"]
