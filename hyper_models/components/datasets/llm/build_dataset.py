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

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from hyper_models.components.datasets.contracts import SampleTransform
from hyper_models.components.datasets.llm.build_indexed_dataset import build_indexed_dataset
from hyper_models.components.datasets.llm.online_dataset import build_online_dataset
from hyper_models.components.datasets.llm.transform_dataset import apply_llm_data_transform
from hyper_models.components.datasets.parallel import DatasetParallelContext

LLMSourceType = Literal["online", "offline"]


def build_llm_dataset(
        *,
        data_config: Mapping[str, Any],
        data_path: str | Sequence[str] | None = None,
        transform: SampleTransform | None = None,
        parallel_context: DatasetParallelContext | None = None,
        tokenizer: Any = None,
        train_valid_test_num_samples: Sequence[int] | None = None,
) -> Any:
    """Build an LLM dataset from its source and transform configuration.

    This function selects only the source reader. Both source types then enter
    the same transform Dataset stage.

    Args:
        data_path: Optional Online local path or required Offline indexed path.
        data_config: Source type and all source-specific build options.
        transform: Callable built by ``Trainer._build_data_transform()``.
            Online uses a tokenizer/chat-template transform; Offline uses a
            pretokenized-field transform.
        parallel_context: Optional distributed construction context for
            indexed Datasets.
        tokenizer: Runtime tokenizer required by offline GPT Datasets.
        train_valid_test_num_samples: Trainer-derived indexed Dataset target sizes.

    Returns:
        A dataset, or the train/validation/test datasets produced by the
        selected source builder.

    Raises:
        ValueError: If ``source_type`` is missing or unsupported.
    """
    try:
        source_type = data_config["source_type"]
    except KeyError as error:
        raise ValueError("data_config must contain 'source_type'") from error

    if source_type == "offline":
        if data_path is None:
            raise ValueError("Offline LLM Datasets require data_path")
        if train_valid_test_num_samples is None:
            raise ValueError("Offline indexed Datasets require Trainer-derived sample counts")
        indexed_data_config = dict(data_config)
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
            data_config=data_config,
            parallel_context=parallel_context,
        )
    else:
        raise ValueError(f"Unsupported LLM source type: {source_type!r}")
    skip_invalid_samples = source_type == "online"
    dataset = apply_llm_data_transform(
        raw_dataset,
        transform,
        skip_invalid_samples=skip_invalid_samples,
    )
    return dataset


__all__ = ["build_llm_dataset"]
