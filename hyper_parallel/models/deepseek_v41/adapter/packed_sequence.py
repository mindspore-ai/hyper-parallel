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
"""Compact Online-packing metadata for DeepSeek-V4.1 CSA2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence,
)
from hyper_parallel.data.batching.attention_runtime import AttentionRuntimeAdapter


class DeepseekV41AttentionRuntimeAdapter(AttentionRuntimeAdapter):
    """Translate global sample boundaries into one contiguous CP shard."""

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
    ) -> SharedCompressedPackedSequence:
        """Build compact boundaries without an O(sequence squared) mask."""
        del sliding_window
        if not causal:
            raise ValueError("DeepSeek-V4.1 CSA2 packed attention must be causal")
        if cp_algorithm != "colossal":
            raise ValueError(
                "DeepSeek-V4.1 CSA2 requires contiguous Colossal CP shards, "
                f"got cp_algorithm={cp_algorithm!r}"
            )
        if len(local_input_shape) != 2 or int(local_input_shape[0]) != 1:
            raise ValueError(
                "DeepSeek-V4.1 compact packing requires local input shape [1, sequence], "
                f"got {tuple(local_input_shape)}"
            )
        local_sequence_length = int(local_input_shape[1])
        return SharedCompressedPackedSequence(
            cu_seq_lens=cu_seq_lens,
            local_query_start=cp_rank * local_sequence_length,
            local_query_length=local_sequence_length,
            global_sequence_length=cp_size * local_sequence_length,
        )

__all__ = ["DeepseekV41AttentionRuntimeAdapter"]
