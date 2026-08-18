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
"""LLM dataset selection and composition."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from hyper_models.components.datasets.contracts import SampleTransform
from hyper_models.components.datasets.llm.build_indexed_dataset import build_indexed_dataset
from hyper_models.components.datasets.llm.online_dataset import build_online_dataset
from hyper_models.components.datasets.llm.transform_dataset import apply_llm_data_transform
from hyper_models.components.datasets.parallel import DatasetParallelContext, create_dataset_parallel_context

logger = logging.getLogger(__name__)


def _get_train_valid_test_num_samples(training_config: Any) -> tuple[int, int, int]:
    """Calculate indexed dataset sizes."""
    if training_config is None:
        raise ValueError("Offline indexed Datasets require a training configuration")

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
    sizes = (train_samples, valid_iters * global_batch_size, eval_iters * global_batch_size)
    logger.debug("Dataset target sizes: train=%d, validation=%d, test=%d", *sizes)
    return sizes


def _build_dataset_parallel_context(
    mesh_context: Any,
    data_config: Mapping[str, Any],
) -> DatasetParallelContext:
    """Build the dataset parallel context."""
    return create_dataset_parallel_context(
        mesh_context,
        data_index_cache=bool(data_config.get("data_index_cache", False)),
        shared_storage=not bool(data_config.get("no_shared_storage", False)),
    )


def build_llm_dataset(
    *,
    data_config: Mapping[str, Any],
    data_path: str | Sequence[str] | None = None,
    transform: SampleTransform | None = None,
    parallel_context: DatasetParallelContext | None = None,
    tokenizer: Any = None,
    train_valid_test_num_samples: Sequence[int] | None = None,
    mesh_context: Any = None,
    training_config: Any = None,
) -> Any:
    """Build and transform an online or offline LLM dataset.

    Args:
        data_config: Source and build options.
        data_path: Online path or offline indexed-data path.
        transform: Sample transform.
        parallel_context: Distributed dataset context.
        tokenizer: Tokenizer used by offline GPT datasets.
        train_valid_test_num_samples: Explicit split sizes.
        mesh_context: Mesh used to build the parallel context.
        training_config: Training plan used to calculate split sizes.

    Returns:
        A dataset or a train/validation/test tuple.

    Raises:
        ValueError: If the source configuration is invalid.
    """
    try:
        source_type = data_config["source_type"]
    except KeyError as error:
        raise ValueError("data_config must contain 'source_type'") from error

    dataset_config = dict(data_config)
    training_seed = getattr(training_config, "seed", None)
    dataset_config["random_seed"] = 42 if training_seed is None else int(training_seed)

    if parallel_context is None and mesh_context is not None:
        parallel_context = _build_dataset_parallel_context(mesh_context, dataset_config)

    logger.debug("Building LLM Dataset: source_type=%s, data_path=%s", source_type, data_path)
    if source_type == "offline":
        if data_path is None and not bool(dataset_config.get("mock_data", False)):
            raise ValueError("Offline LLM Datasets require data_path")

        if train_valid_test_num_samples is None:
            train_valid_test_num_samples = _get_train_valid_test_num_samples(training_config)

        indexed_data_config = dict(dataset_config)
        if tokenizer is not None:
            indexed_data_config["tokenizer"] = tokenizer
        raw_dataset = build_indexed_dataset(
            data_path=data_path,
            data_config=indexed_data_config,
            train_valid_test_num_samples=train_valid_test_num_samples,
            parallel_context=parallel_context,
        )
    elif source_type == "online":
        raw_dataset = build_online_dataset(
            data_path=data_path,
            data_config=dataset_config,
            parallel_context=parallel_context,
        )
    else:
        raise ValueError(f"Unsupported LLM source type: {source_type!r}")
    dataset = apply_llm_data_transform(raw_dataset, transform, skip_invalid_samples=source_type == "online")
    logger.debug("Built LLM Dataset: source_type=%s, result_type=%s", source_type, type(dataset).__name__)
    return dataset


__all__ = ["build_llm_dataset"]
