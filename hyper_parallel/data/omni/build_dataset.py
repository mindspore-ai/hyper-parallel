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
"""Local Omni sources and encoding hooks, independent of text dataset builders."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from typing import Any

# Match the existing Torch StatefulDataLoader; this is not a core backend API.
from torch.utils.data import Dataset  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.omni.constants import ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.omni.omni_transform import OmniDataTransform, _OmniTransformStrategy
from hyper_parallel.data.parallel import create_dataloader_parallel_context


class _ParquetRecords:
    """Index rows without decoding media; cache at most one row group per worker."""

    def __init__(self, path: str, transform: OmniDataTransform) -> None:
        """Scan bounded batches for valid row indices without retaining open files."""
        # Parquet is optional; JSON/text users must not require Arrow at import time.
        from pyarrow import parquet  # pylint: disable=C0415

        self.path = path
        self.indices = []
        self._cache_key = None
        self._table = None
        with parquet.ParquetFile(path) as source:
            for group in range(source.num_row_groups):
                offset = 0
                for batch in source.iter_batches(batch_size=256, row_groups=[group]):
                    for index, record in enumerate(batch.to_pylist(), start=offset):
                        if transform.is_valid_sample(record):
                            self.indices.append((group, index))
                    offset += batch.num_rows

    def __getstate__(self) -> dict:
        """Avoid serializing Arrow caches into spawned DataLoader workers."""
        state = self.__dict__.copy()
        state.update(_cache_key=None, _table=None)
        return state

    def __getitem__(self, index: tuple[int, int]) -> dict:
        """Return a fresh Python record from the worker-local row-group cache."""
        # Lazy optional import also permits spawned workers to reconstruct this reader.
        from pyarrow import parquet  # pylint: disable=C0415

        group, row = index
        cache_key = (os.getpid(), group)
        if self._cache_key != cache_key:
            self._table = None
            with parquet.ParquetFile(self.path) as source:
                self._table = source.read_row_group(group)
            self._cache_key = cache_key
        return self._table.slice(row, 1).to_pylist()[0]


class OmniMappingDataset(Dataset):
    """Read JSON/JSONL/Parquet records and defer encoding to the configured lifecycle.

    JSONL uses byte offsets so heterogeneous tool/media records do not need
    Arrow schema unification. Source order is left to the existing DP sampler.
    """

    def __init__(self, data_path: str, transform: OmniDataTransform) -> None:
        """Index local records without eagerly decoding their media."""
        self.data_path = os.path.abspath(os.fspath(data_path))
        self.data_transform = transform
        self.strategy = _OmniTransformStrategy.from_transform(transform)
        self.records = None
        self.parquet_records = None
        self.indices = []
        if self.data_path.endswith(".jsonl"):
            with open(self.data_path, "rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip() and transform.is_valid_sample(json.loads(line)):
                        self.indices.append(offset)
        elif self.data_path.endswith(".json"):
            with open(self.data_path, encoding="utf-8") as handle:
                self.records = json.load(handle)
            if not isinstance(self.records, list):
                raise ValueError("Omni JSON must contain a list of records")
            self.indices = [
                index for index, record in enumerate(self.records) if transform.is_valid_sample(record)
            ]
        elif self.data_path.endswith(".parquet"):
            self.parquet_records = _ParquetRecords(self.data_path, transform)
            self.indices = self.parquet_records.indices
        else:
            raise ValueError("Omni sources support local .json, .jsonl or .parquet files")
        if not self.indices:
            raise ValueError("Omni source contains no valid records")

    def __len__(self) -> int:
        """Return the number of valid source records."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        """Prepare one candidate without consuming other source records."""
        source_index = self.indices[index]
        if self.parquet_records is not None:
            record = self.parquet_records[source_index]
        elif self.records is None:
            with open(self.data_path, "rb") as handle:
                handle.seek(source_index)
                record = json.loads(handle.readline())
        else:
            record = copy.deepcopy(self.records[source_index])
        record[ONLINE_SOURCE_PATH_KEY] = self.data_path
        return self.strategy.prepare_sample(record)

    def encode_selected_sample(self, sample: Any) -> Any:
        """Finish encoding only a selected candidate.

        Args:
            sample: Selected candidate.
        """
        return self.strategy.encode_selected_sample(sample)

    def encode_batch(self, batch: Any) -> Any:
        """Adapt collated fields to the model-specific forward contract.

        Args:
            batch: Collated model inputs.
        """
        return self.data_transform.encode_batch(batch)


def build_online_omni_mapping_dataset(
        *,
        data_path: str | Mapping[str, str],
        transform: OmniDataTransform,
        data_config: Mapping[str, Any] | None = None,
        mesh_context: Any = None,
) -> Any:
    """Build local multimodal datasets without replacing text source policy.

    Args:
        data_path: Local JSON/JSONL/Parquet file or explicit train/valid/test path mapping.
        transform: Omni encoding lifecycle.
        data_config: Reserved options; nonempty options are rejected.
        mesh_context: Optional trainer mesh, used to build only on TP source ranks.

    Returns:
        One dataset, an explicit split tuple, or None on a non-source TP rank.
    """
    if data_config:
        raise ValueError("Use explicit split paths; isolated Omni does not implement source blending options")
    if not isinstance(transform, OmniDataTransform):
        raise TypeError("Omni dataset requires an OmniDataTransform")
    if mesh_context is not None:
        context = create_dataloader_parallel_context(mesh_context)
        if not context.build_on_rank():
            return None
    if isinstance(data_path, Mapping):
        unknown = set(data_path) - {"train", "valid", "test"}
        if unknown or "train" not in data_path:
            raise ValueError("Omni split paths require train and optionally valid/test")
        return tuple(
            OmniMappingDataset(data_path[name], transform) if name in data_path else None
            for name in ("train", "valid", "test")
        )
    return OmniMappingDataset(data_path, transform)
