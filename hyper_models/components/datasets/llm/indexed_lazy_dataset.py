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

from collections.abc import Callable, Mapping
from typing import Any


class LazyDatasetProxy:
    """Construct and cache a Dataset when it is first accessed."""

    def __init__(
        self,
        dataset_factory: Callable[[], Any],
        *,
        dataset_length: int,
        unique_identifiers: Any,
    ) -> None:
        """Store lightweight metadata without constructing the Dataset.

        Args:
            dataset_factory: Callable that constructs the real Dataset on first
                sample access.
            dataset_length: Number of samples exposed by the real Dataset.
            unique_identifiers: Stable Dataset cache identity assembled from
                construction inputs.
        """
        self._dataset_factory = dataset_factory
        self._dataset_length = dataset_length
        self._unique_identifiers = unique_identifiers
        self._dataset = None

    def __len__(self) -> int:
        """Return the known Dataset length without constructing it."""
        return self._dataset_length

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Construct the Dataset if necessary and read one sample."""
        dataset = self._get_dataset()
        sample = dataset[index]
        return sample

    @property
    def unique_identifiers(self) -> Any:
        """Return the known identity without constructing the Dataset."""
        return self._unique_identifiers

    def _get_dataset(self) -> Any:
        """Construct the Dataset once and reuse it for later accesses."""
        if self._dataset is None:
            self._dataset = self._dataset_factory()
        return self._dataset
