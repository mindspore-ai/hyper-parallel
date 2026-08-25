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
"""Resolve global sequence boundaries from Online or Indexed batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.platform import get_platform


platform = get_platform()


class OnlineBoundaryResolver:
    """Read global cumulative sequence boundaries emitted by Online packing."""

    def resolve(self, canonical_batch: Mapping[str, Any]) -> Any:
        """Read leading-zero ``cu_seq_lens`` from an Online batch.

        Args:
            canonical_batch: Normalized batch produced by the Online packing collator.

        Returns:
            Global int32 cumulative sequence boundaries.
        """
        raw_cu_seq_lens = canonical_batch["cu_seq_lens"]
        cu_seq_lens = platform.tensor_type_cast(raw_cu_seq_lens, "int32")

        return cu_seq_lens


class IndexedBoundaryResolver:
    """Recover global cumulative sequence boundaries from Indexed input IDs."""

    def __init__(self, eod_token_id: int | None) -> None:
        """Initialize the Indexed boundary resolver.

        Args:
            eod_token_id: Token separating packed Indexed sequences.
        """
        self.eod_token_id = eod_token_id

    def resolve(self, canonical_batch: Mapping[str, Any]) -> Any:
        """Recover leading-zero ``cu_seq_lens`` from Indexed tokens.

        Args:
            canonical_batch: Normalized Indexed batch containing input IDs.

        Returns:
            Global int32 cumulative sequence boundaries.
        """
        input_ids = canonical_batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        token_indices = platform.arange(seq_len, dtype=input_ids.dtype, device=input_ids.device)

        # Collect flattened, nonzero cumulative sequence ends. A leading zero
        # is prepended below to form ``cu_seq_lens``.
        seq_ends = []
        for batch_idx in range(batch_size):
            if self.eod_token_id is None:
                eod_indices = token_indices[:0]
            else:
                eod_indices = token_indices[input_ids[batch_idx] == self.eod_token_id]

            prev_eod_idx = -1
            for eod_idx in eod_indices:
                eod_idx = int(eod_idx.item())
                # Stop boundary recovery when consecutive EOD tokens appear.
                if eod_idx == prev_eod_idx:
                    break

                seq_end = batch_idx * seq_len + eod_idx + 1
                seq_ends.append(seq_end)
                prev_eod_idx = eod_idx + 1

            # Every DataLoader row remains a complete sequence boundary.
            row_end = (batch_idx + 1) * seq_len
            if not seq_ends or seq_ends[-1] != row_end:
                seq_ends.append(row_end)

        # Packed attention uses the standard leading-zero cumulative form.
        cu_seq_lens = platform.tensor([0, *seq_ends], dtype=platform.tensor_dtype.int32)

        return cu_seq_lens
