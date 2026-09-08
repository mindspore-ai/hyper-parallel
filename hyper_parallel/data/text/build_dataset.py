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
"""Public Online and Indexed text Dataset builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.text.build_data_transform import PlaintextTransform
from hyper_parallel.data.indexed.indexed_dataset import (
    build_indexed_dataset as _build_indexed_dataset,
)
from hyper_parallel.data.text.online.online_dataset import (
    build_online_dataset as _build_online_dataset,
)
from hyper_parallel.data.text.online.online_utils import ONLINE_PLAINTEXT_TEXT_KEYS_KEY
from hyper_parallel.data.text.transform_dataset import apply_llm_data_transform
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext,
    create_dataloader_parallel_context,
)

logger = get_dataset_logger(__name__)


def _build_dataloader_context(
        mesh_context: Any,
        data_config: Mapping[str, Any],
) -> DataLoaderParallelContext | None:
    """Build DataLoader ownership from the runtime mesh when available."""
    if mesh_context is None:
        return None

    dataloader_context = create_dataloader_parallel_context(
        mesh_context,
        data_index_cache=bool(data_config.get("data_index_cache", False)),
        shared_storage=not bool(data_config.get("no_shared_storage", False)),
    )
    return dataloader_context


def _get_indexed_split_sizes(training_config: Any) -> tuple[int, int, int]:
    """Calculate Indexed Dataset target sizes from the training plan."""
    if training_config is None:
        raise ValueError("Indexed Dataset requires a training configuration")

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
    logger.debug("Indexed Dataset target sizes: train=%d, validation=%d, test=%d", *split_sizes)
    return split_sizes


def build_online_text_dataset(
        *,
        data_config: Mapping[str, Any],
        data_path: str | Sequence[str] | None = None,
        transform: Callable[[Any], Any] | None = None,
        dataloader_context: DataLoaderParallelContext | None = None,
        mesh_context: Any = None,
        training_config: Any = None,
) -> Any:
    """Build an Online source and apply its text transform.

    Args:
        data_config: Online mapping or iterable source options.
        data_path: Optional local source path or ordered paths.
        transform: Plaintext or conversation sample transform.
        dataloader_context: Optional explicit DataLoader ownership context.
        mesh_context: Runtime mesh used to derive DataLoader ownership.
        training_config: Training plan providing the random seed.

    Returns:
        A transformed Online Dataset on each DataLoader-owning rank.

    Raises:
        ValueError: If no Online text transform is configured.
    """
    if transform is None:
        raise ValueError("Online Dataset requires a plaintext or conversation data_transform")

    dataset_config = dict(data_config)
    training_seed = getattr(training_config, "seed", None)
    dataset_config["random_seed"] = 42 if training_seed is None else int(training_seed)
    if isinstance(transform, PlaintextTransform):
        dataset_config[ONLINE_PLAINTEXT_TEXT_KEYS_KEY] = transform.text_keys
    if dataloader_context is None:
        dataloader_context = _build_dataloader_context(mesh_context, dataset_config)

    online_dataset = _build_online_dataset(
        data_path=data_path,
        data_config=dataset_config,
        dataloader_context=dataloader_context,
    )
    transformed_dataset = apply_llm_data_transform(
        online_dataset,
        transform,
        skip_invalid_samples=True,
    )
    return transformed_dataset


def build_indexed_text_dataset(
        *,
        data_config: Mapping[str, Any],
        data_path: str | Sequence[str] | None = None,
        tokenizer: Any = None,
        train_valid_test_num_samples: Sequence[int] | None = None,
        dataloader_context: DataLoaderParallelContext | None = None,
        mesh_context: Any = None,
        training_config: Any = None,
) -> Any:
    """Build train, validation, and test Datasets from Indexed token files.

    Args:
        data_config: Indexed Dataset and split options.
        data_path: Indexed prefix, directory, or ordered paths.
        tokenizer: Tokenizer providing vocabulary and EOD metadata.
        train_valid_test_num_samples: Optional explicit split sizes.
        dataloader_context: Optional explicit DataLoader ownership context.
        mesh_context: Runtime mesh used to derive DataLoader ownership.
        training_config: Training plan used to derive split sizes and seed.

    Returns:
        Train, validation, and test Indexed Datasets.

    Raises:
        ValueError: If required path or training information is missing.
    """
    dataset_config = dict(data_config)
    training_seed = getattr(training_config, "seed", None)
    dataset_config["random_seed"] = 42 if training_seed is None else int(training_seed)
    if tokenizer is not None:
        dataset_config["tokenizer"] = tokenizer

    if data_path is None and not bool(dataset_config.get("mock_data", False)):
        raise ValueError("Indexed Dataset requires data_path")

    if train_valid_test_num_samples is None:
        train_valid_test_num_samples = _get_indexed_split_sizes(training_config)

    if dataloader_context is None:
        dataloader_context = _build_dataloader_context(mesh_context, dataset_config)

    indexed_datasets = _build_indexed_dataset(
        data_path=data_path,
        data_config=dataset_config,
        train_valid_test_num_samples=train_valid_test_num_samples,
        dataloader_context=dataloader_context,
    )
    return indexed_datasets
