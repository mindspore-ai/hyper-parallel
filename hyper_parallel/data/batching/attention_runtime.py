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
"""Adapt compact sequence boundaries to an Attention/CP runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from hyper_parallel.platform import get_platform


platform = get_platform()


def build_dense_attention_masks(
        *,
        cu_seq_lens: Any,
        micro_batch_size: int,
        seq_length: int,
        device: Any,
        reset_attention_mask: bool,
        sliding_window: int | None,
) -> tuple[Any, Any | None]:
    """Build global dense attention and optional sliding-window masks.

    A boolean ``True`` marks an allowed query-key pair. Packed boundaries use
    flattened ``[B, S]`` offsets and remain global across CP ranks. The final
    boundary must cover the physical sequence length, including alignment
    padding represented as a synthetic final sequence.
    """
    boundaries = [int(boundary) for boundary in cu_seq_lens.tolist()]
    expected_total = micro_batch_size * seq_length
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError("cu_seq_lens must contain a leading zero and at least one sequence")
    if any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("cu_seq_lens must be strictly increasing")
    if boundaries[-1] != expected_total:
        raise ValueError(
            f"cu_seq_lens must cover the physical batch length ({expected_total}), but got {boundaries[-1]}"
        )

    # Attention mask (lower triangular).
    att_mask_batch = micro_batch_size if reset_attention_mask else 1
    attention_mask = platform.ones((att_mask_batch, seq_length, seq_length), dtype=bool, device=device).tril()
    attention_mask = attention_mask.view(att_mask_batch, 1, seq_length, seq_length)

    if reset_attention_mask:
        for seq_start in boundaries[:-1]:
            batch_idx, row_seq_start = divmod(seq_start, seq_length)
            # Tokens after this boundary cannot attend to an earlier sequence.
            # [B, 1, S, S]， batch，head，query, key
            attention_mask[batch_idx, 0, row_seq_start:, :row_seq_start] = False

    swa_mask = None
    if sliding_window is not None:
        positions = platform.arange(seq_length, dtype=platform.tensor_dtype.int64, device=device)
        query_positions = positions.unsqueeze(1)
        key_positions = positions.unsqueeze(0)
        token_distance = query_positions - key_positions
        outside_window = token_distance > sliding_window
        swa_mask = attention_mask & ~outside_window.unsqueeze(0).unsqueeze(0)

    return attention_mask, swa_mask


class AttentionRuntimeAdapter(ABC):
    """Build backend-owned metadata for compressed packed attention.

    Dense attention bypasses this interface and materializes its attention
    masks locally. A compressed backend derives FA and CP runtime metadata
    from the global sequence boundaries without changing the batch contract.
    The concrete adapter owns backend-specific physical layouts such as
    ``thd`` or ``TND``.
    """

    @abstractmethod
    def build_packed_seq_params(
            self,
            *,
            cu_seq_lens: Any,
            local_input_shape: Sequence[int],
            cp_rank: int,
            cp_size: int,
            cp_algorithm: str,
            causal: bool,
            sliding_window: int | None,
    ) -> object:
        """Build packed-attention metadata for the current parallel rank.

        Args:
            cu_seq_lens: Global cumulative sequence boundaries with a leading
                zero. A synthetic final sequence covers alignment padding, so
                the final boundary equals the physical global token count. The
                boundaries remain unsharded across CP ranks.
            local_input_shape: Shape of the CP-local input IDs.
            cp_rank: Rank in the context-parallel group.
            cp_size: Number of ranks in the context-parallel group.
            cp_algorithm: Context-parallel algorithm selecting local Q/KV data.
            causal: Whether the attention computation is causal.
            sliding_window: Sliding-window size, or ``None`` for full attention.

        Returns:
            Backend-owned packed sequence parameters passed to model attention.

        Note:
            Implementations derive Q/KV cumulative lengths and maximum sequence
            lengths for every backend. CP algorithms may additionally provide
            local Q/KV boundaries, slice indexes, and communication split sizes.
        """
        raise NotImplementedError
