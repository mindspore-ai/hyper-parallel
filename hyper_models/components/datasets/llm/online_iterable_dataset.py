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
"""VeOmni-style Online iterable Dataset source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyper_models.components.datasets.llm.online_utils import (
    load_online_hf_dataset,
    normalize_online_parallel_context,
    split_online_dataset_by_dp,
)
from hyper_models.components.datasets.parallel import DatasetParallelContext, build_distributed_dataset


def build_online_iterable_dataset(
        *,
        data_config: Mapping[str, Any],
        data_path: str | Sequence[str] | None = None,
        parallel_context: DatasetParallelContext | None = None,
) -> Any:
    """Build a shuffled, DP-sharded, stateful Online raw-record stream.

    Args:
        data_path: Optional local JSON/JSONL/Parquet/CSV/Arrow paths.
        data_config: Streaming options including ``seed``, ``shuffle``,
            ``shuffle_buffer_size``, and ``split_by_data_parallel``.
        parallel_context: Dataset ownership and DP topology.

    Returns:
        A Hugging Face iterable Dataset on TP rank zero, otherwise ``None``.
    """
    dataset_context = normalize_online_parallel_context(parallel_context)

    def dataset_factory() -> Any:
        """Load, shuffle, and DP-shard the upstream stream."""
        online_dataset = load_online_hf_dataset(
            data_path=data_path,
            data_config=data_config,
            streaming=True,
        )
        if bool(data_config.get("shuffle", True)):
            random_seed = int(data_config.get("seed", 42))
            buffer_size = int(data_config.get("shuffle_buffer_size", 10_000))
            if buffer_size <= 0:
                raise ValueError("Online shuffle_buffer_size must be positive")
            online_dataset = online_dataset.shuffle(
                seed=random_seed,
                buffer_size=buffer_size,
            )
        if bool(data_config.get("split_by_data_parallel", True)):
            online_dataset = split_online_dataset_by_dp(online_dataset, dataset_context)
        return online_dataset

    online_dataset = build_distributed_dataset(
        dataset_factory,
        dataset_context,
        barrier_needed=False,
    )
    return online_dataset
