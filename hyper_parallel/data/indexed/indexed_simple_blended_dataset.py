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
"""Index-free blending for pre-shuffled MR Datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

SimpleBlendMode = Literal["inter", "intra"]


class SimpleBlendedDataset:
    """Blend equally weighted MR Datasets in interleaved or contiguous order."""

    def __init__(
            self,
            datasets: Sequence[Any],
            size: int,
            mode: SimpleBlendMode,
    ) -> None:
        """Validate the inputs and create the configured simple ordering."""
        if not datasets:
            raise ValueError("simple blend requires at least one Dataset")
        self.datasets = list(datasets)
        self.size = size
        self.mode = mode
        self._locations = self._build_locations()
        if self.size > len(self._locations):
            raise ValueError(
                f"Requested {self.size} samples from a simple blend containing "
                f"{len(self._locations)} samples"
            )

    def __len__(self) -> int:
        """Return the requested blended sample count."""
        return self.size

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Read one MR sample in interleaved or Dataset-contiguous order."""
        dataset_id, sample_id = self._locations[index]
        sample = {"dataset_id": dataset_id, **self.datasets[dataset_id][sample_id]}
        return sample

    def _build_locations(self) -> list[tuple[int, int]]:
        """Map blend positions to Dataset and sample indices."""
        if self.mode == "intra":
            locations = [
                (dataset_id, sample_id)
                for dataset_id, dataset in enumerate(self.datasets)
                for sample_id in range(len(dataset))
            ]
            return locations

        # mode = inter
        locations = self._build_interleaved_locations()
        return locations

    def _build_interleaved_locations(self) -> list[tuple[int, int]]:
        """Alternate over non-empty Datasets until every sample is exposed."""
        max_size = max(len(dataset) for dataset in self.datasets)
        locations = [
            (dataset_id, sample_id)
            for sample_id in range(max_size)
            for dataset_id, dataset in enumerate(self.datasets)
            if sample_id < len(dataset)
        ]
        return locations
