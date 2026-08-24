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
"""Dispatch the reserved online LLM Dataset implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hyper_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_models.components.datasets.llm.online_iterable_dataset import build_online_iterable_dataset
from hyper_models.components.datasets.llm.online_mapping_dataset import build_online_mapping_dataset
from hyper_models.components.datasets.parallel import DatasetParallelContext

logger = get_dataset_logger(__name__)

_ONLINE_DATASET_BUILDERS: dict[str, Callable[..., Any]] = {
    "mapping": build_online_mapping_dataset,
    "iterable": build_online_iterable_dataset,
}


def build_online_dataset(
        *,
        data_config: Mapping[str, Any],
        data_path: str | Sequence[str] | None = None,
        parallel_context: DatasetParallelContext | None = None,
) -> Any:
    """Select an online Dataset interface that produces RawSamples.

    Args:
        data_path: Optional local source path or ordered paths.
        data_config: Online options containing ``dataset_type``.
        parallel_context: Optional Dataset rank ownership policy.

    Returns:
        The mapping or iterable online Dataset when its implementation exists.

    Raises:
        ValueError: If ``dataset_type`` is unsupported.
        NotImplementedError: From the selected reserved builder.
    """
    dataset_type = str(data_config.get("dataset_type", "mapping"))
    try:
        online_dataset_builder = _ONLINE_DATASET_BUILDERS[dataset_type]
    except KeyError as error:
        supported_types = sorted(_ONLINE_DATASET_BUILDERS)
        raise ValueError(
            f"Unsupported online dataset_type {dataset_type!r}; expected one of {supported_types!r}"
        ) from error
    logger.debug("Selected online Dataset type=%s, data_path=%s", dataset_type, data_path)
    online_dataset = online_dataset_builder(
        data_path=data_path,
        data_config=data_config,
        parallel_context=parallel_context,
    )
    logger.debug("Built online Dataset type=%s", type(online_dataset).__name__)
    return online_dataset
