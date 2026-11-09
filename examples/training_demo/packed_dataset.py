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
"""Read offline-packed Arrow records with explicit independent sequence boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk
import numpy as np


class PackedTextDataset:
    """Expose pre-shifted token targets and boundaries for native sequence repacking."""

    def __init__(
            self,
            data_path: str | Path,
            sequence_length: int,
            data_config: Mapping[str, Any] | None = None,
            **_: Any,
    ) -> None:
        """Open a Hugging Face Dataset saved with ``save_to_disk``.

        Args:
            data_path: Saved Dataset directory containing input_ids, labels and cu_seqlens.
            sequence_length: Maximum packed record length and target packing capacity.
            data_config: Options consumed by the shared Trainer's native sampler loader.
        """
        self._records = load_from_disk(str(data_path))
        if not isinstance(self._records, Dataset):
            raise ValueError("Packed data_path must select one Dataset, not a DatasetDict")
        required = {"input_ids", "labels", "cu_seqlens"}
        if not required.issubset(self._records.column_names):
            raise ValueError(f"Packed Dataset requires columns {sorted(required)}")
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        self._sequence_length = sequence_length
        self.data_config = dict(data_config or {})

    def __len__(self) -> int:
        """Return the original packed-record count used by the native sampler."""
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Read one complete record without changing labels or sequence boundaries.

        Args:
            index: Original packed-record index selected by the native sampler.

        Returns:
            Arrays in the shared packed-text field format.
        """
        record = self._records[int(index)]
        input_ids = np.asarray(record["input_ids"], dtype=np.int64)
        labels = np.asarray(record["labels"], dtype=np.int64)
        if input_ids.ndim != 1 or not 0 < input_ids.size <= self._sequence_length:
            raise ValueError("Packed record must be a nonempty vector within sequence_length")
        if labels.shape != input_ids.shape:
            raise ValueError("Packed labels must align with input_ids")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "cu_seq_lens": np.asarray(record["cu_seqlens"]),
        }
