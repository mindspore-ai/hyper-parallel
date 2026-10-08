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
"""Pre-tokenized supervised records stored in aligned Megatron indexed streams."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader


class IndexedSupervisedDataset:
    """Read tokens and pre-shifted labels, folding a 0/1 loss mask into the labels.

    A prefix identifies three standard .bin/.idx pairs: ``.tokens``, ``.labels``
    and ``.loss_mask``. Record order and boundaries must match across streams.
    Targets whose mask is 0 become ``IGNORE_INDEX``, so the shared text batch
    derives the loss mask from the labels as for every other text model.
    Tokenization, shifting and supervision selection happen during preparation;
    the shared DataLoader owns sampling and batching.
    """

    def __init__(self, data_path: str | Path, sequence_length: int | None = None) -> None:
        """Open aligned indexed streams and validate their record metadata.

        Args:
            data_path: Common prefix before .tokens/.labels/.loss_mask.
            sequence_length: Optional required length of every complete record.
        """
        self.readers = {name: IndexedDataReader(f"{data_path}.{name}")
                        for name in ("tokens", "labels", "loss_mask")}
        index = self.readers["tokens"].index
        if len(index) == 0 or np.any(index.sequence_lengths <= 0):
            raise ValueError("Supervised indexed data requires nonempty records")
        if sequence_length is not None and np.any(index.sequence_lengths != sequence_length):
            raise ValueError("Supervised indexed record length does not match sequence_length")
        for name, reader in self.readers.items():
            if not np.array_equal(index.sequence_lengths, reader.index.sequence_lengths):
                raise ValueError(f"Supervised {name} record lengths do not match tokens")
            if not np.array_equal(index.document_indices, reader.index.document_indices):
                raise ValueError(f"Supervised {name} document boundaries do not match tokens")
            if name != "loss_mask" and not np.issubdtype(reader.index.dtype, np.integer):
                raise ValueError(f"Supervised {name} must use an integer indexed dtype")

    def __len__(self) -> int:
        """Return the number of aligned records."""
        return len(self.readers["tokens"])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Return tokens and labels; targets with a zero mask become ``IGNORE_INDEX``.

        Args:
            index: Record index selected by the DataLoader sampler.
        """
        sample = {name: np.array(reader[index], dtype=np.float32 if name == "loss_mask" else np.int64,
                                 copy=True) for name, reader in self.readers.items()}
        mask = sample.pop("loss_mask")
        if np.any(sample["tokens"] < 0):
            raise ValueError("Input tokens must be nonnegative")
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError("Loss mask values must be 0 or 1")
        if np.any((sample["labels"] < 0) & (mask != 0)):
            raise ValueError("Ignored labels must have zero loss weight")
        sample["labels"][mask == 0] = IGNORE_INDEX
        return sample


class JTSequenceRuntime(RuntimeInputAdapter):
    """Pass the public batch's document boundaries to JT without constructing a dense mask."""

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the model-owned cumulative-length input."""
        return ("actual_seq_len",)

    def build_runtime_inputs(self, *, batch: Mapping[str, Any],
                             context: RuntimeInputContext) -> Mapping[str, Any]:
        """Convert global leading-zero boundaries once at the data/model boundary.

        Args:
            batch: Local tokens and global cumulative document boundaries.
            context: Local sequence shape and CP degree from the public data path.
        """
        boundaries = batch.get("cu_seq_lens")
        if boundaries is None:
            return {}
        values = tuple(int(value) for value in boundaries.tolist())
        global_length = context.local_input_shape[1] * context.parallel_sizes["cp"]
        if (context.local_input_shape[0] != 1 or len(values) < 2 or values[0] != 0
                or values[-1] != global_length
                or any(left >= right for left, right in zip(values, values[1:]))):
            raise ValueError("JT packed boundaries must cover one global batch-one sequence")
        return {"actual_seq_len": values[1:]}
