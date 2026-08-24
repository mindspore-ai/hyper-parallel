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
"""VeOmni-style Online mapping Dataset source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from hyper_models.components.distributed.infrastructure import OnlineDatasetBarrier
from hyper_models.components.datasets.contracts import RawSample
from hyper_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_models.components.datasets.llm.online_utils import (
    load_online_hf_dataset,
    normalize_online_parallel_context,
)
from hyper_models.components.datasets.parallel import DatasetParallelContext, build_distributed_dataset

logger = get_dataset_logger(__name__)


class OnlineMappingDataset:
    """Expose Hugging Face records through deterministic integer indices."""

    def __init__(self, source_dataset: Any) -> None:
        """Store the raw mapping Dataset without applying tokenizer logic."""
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        """Return the finite raw-record count."""
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> RawSample:
        """Read and validate one RawSample."""
        raw_sample = self.source_dataset[index]
        if not isinstance(raw_sample, Mapping):
            raise ValueError("Online mapping source records must be mappings")
        normalized_sample = dict(raw_sample)
        return normalized_sample


def build_online_mapping_dataset(
        *,
        data_config: Mapping[str, Any],
        data_path: str | Sequence[str] | None = None,
        parallel_context: DatasetParallelContext | None = None,
) -> Any:
    """Build a finite Online Dataset that produces text RawSamples.

    Args:
        data_path: Optional local JSON/JSONL/Parquet/CSV/Arrow paths.
        data_config: Namespace/cache options or ``hf_dataset_name``.
        parallel_context: Dataset rank ownership and synchronization policy.

    Returns:
        An Online mapping Dataset on TP rank zero, otherwise ``None``.
    """
    dataset_context = normalize_online_parallel_context(parallel_context)
    if dataset_context.distributed_enabled:
        dataset_context = replace(
            dataset_context,
            barrier=OnlineDatasetBarrier(),
        )

    def dataset_factory() -> OnlineMappingDataset:
        """Load the raw source and attach its integer-index wrapper."""
        source_dataset = load_online_hf_dataset(
            data_path=data_path,
            data_config=data_config,
            streaming=False,
        )
        online_dataset = OnlineMappingDataset(source_dataset)
        logger.debug("Loaded online mapping Dataset records=%d", len(online_dataset))
        return online_dataset

    online_dataset = build_distributed_dataset(
        dataset_factory,
        dataset_context,
        # Cache-builder ranks must finish the Hub download before the other
        # owning ranks reopen the shared Hugging Face cache.
        barrier_needed=dataset_context.distributed_enabled,
    )
    return online_dataset
