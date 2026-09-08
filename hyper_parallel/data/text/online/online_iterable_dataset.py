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
"""Online iterable dataset source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.text.online.online_utils import (
    load_online_hf_dataset,
    normalize_online_dataloader_context,
)
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext,
    build_dataset_for_dataloader,
    split_iterable_dataset_by_dp,
)

logger = get_dataset_logger(__name__)


def build_online_iterable_dataset(
        *,
        data_config: Mapping[str, Any],
        data_path: str | Sequence[str] | None = None,
        dataloader_context: DataLoaderParallelContext | None = None,
) -> Any:
    """Build a shuffled, DP-sharded, stateful Online raw-record stream.

    Args:
        data_path: Optional local JSON/JSONL/Parquet/CSV/Arrow paths.
        data_config: Streaming options including ``shuffle``,
            ``shuffle_buffer_size``, and ``split_by_data_parallel``.
        dataloader_context: DataLoader ownership and DP topology.

    Returns:
        A Hugging Face iterable Dataset on TP rank zero, otherwise ``None``.
    """
    normalized_context = normalize_online_dataloader_context(dataloader_context)

    def dataset_factory() -> Any:
        """Load, shuffle, and DP-shard the upstream stream."""
        # Read samples lazily as a stream; this does not perform DP sharding.
        online_dataset = load_online_hf_dataset(
            data_path=data_path,
            data_config=data_config,
            streaming=True,
        )

        if bool(data_config.get("shuffle", True)):
            random_seed = int(data_config.get("random_seed", 42))
            buffer_size = int(data_config.get("shuffle_buffer_size", 10_000))
            if buffer_size <= 0:
                raise ValueError("Online shuffle_buffer_size must be positive")
            online_dataset = online_dataset.shuffle(
                seed=random_seed,
                buffer_size=buffer_size,
            )
            logger.debug("Enabled online iterable shuffle: seed=%d, buffer_size=%d", random_seed, buffer_size)

        if bool(data_config.get("split_by_data_parallel", True)):
            # Iterable sources have no index sampler, so shard the stream across DP ranks here.
            online_dataset = split_iterable_dataset_by_dp(online_dataset, normalized_context)
            logger.debug(
                "Split online iterable Dataset by DP: rank=%d, world_size=%d",
                normalized_context.dp_rank, normalized_context.dp_world_size,
            )

        return online_dataset

    online_dataset = build_dataset_for_dataloader(
        dataset_factory,
        normalized_context,
        barrier_needed=False,
    )

    return online_dataset
