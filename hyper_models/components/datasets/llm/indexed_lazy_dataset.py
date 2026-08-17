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
"""Lazy proxy for deferred indexed Dataset construction."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)


class LazyDatasetProxy:
    """Construct and cache a Dataset when it is first accessed."""

    def __init__(
            self,
            dataset_factory: Callable[[], Any],
            *,
            unique_identifiers: Any,
    ) -> None:
        """Store lightweight metadata without constructing the Dataset.

        Args:
            dataset_factory: Callable that constructs the real Dataset on first
                sample access.
            unique_identifiers: Stable Dataset cache identity assembled from
                construction inputs.
        """
        self._dataset_factory = dataset_factory
        self._unique_identifiers = unique_identifiers
        self._dataset = None

    def __len__(self) -> int:
        """Initialize the Dataset and return its length."""
        return len(self._get_dataset())

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Construct the Dataset if necessary and read one sample."""
        dataset = self._get_dataset()
        sample = dataset[index]
        return sample

    @property
    def unique_identifiers(self) -> Any:
        """Initialize the Dataset and return its cache identity."""
        return self._get_dataset().unique_identifiers

    def _get_dataset(self) -> Any:
        """Construct the Dataset once and reuse it for later accesses."""
        if self._dataset is None:
            identifiers = self._unique_identifiers if isinstance(self._unique_identifiers, Mapping) else {}
            logger.debug(
                "Initializing lazy Dataset: class=%s, path=%s, split=%s, samples=%s",
                identifiers.get("class"), identifiers.get("dataset_path"), identifiers.get("index_split"),
                identifiers.get("num_samples"),
            )
            self._dataset = self._dataset_factory()
        return self._dataset
