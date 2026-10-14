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
"""DeepSeek V4.1 sparse-attention modules backed by optional FlashMLA."""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.deepseek_ascend import (
    deepseek_flash_mla_metadata,
    deepseek_flash_mla_sparse_decode,
    deepseek_flash_mla_sparse_prefill,
)


class DeepseekV41SparsePrefillAttention(nn.Module):
    """Stateless DeepSeek V4.1 sparse-prefill attention.

    Args:
        softmax_scale: Scale applied to attention logits.
        value_head_dim: Value dimension. FlashMLA currently requires 512.
    """

    def __init__(self, softmax_scale: float, value_head_dim: int = 512) -> None:
        """Initialize sparse-prefill attention configuration."""
        super().__init__()
        _validate_value_head_dim(value_head_dim)
        self.softmax_scale = softmax_scale
        self.value_head_dim = value_head_dim

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        indices: torch.Tensor,
        *,
        attention_sink: torch.Tensor | None = None,
        topk_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply FlashMLA sparse-prefill attention.

        Args:
            query: BF16 query tensor ``[sequence, heads, 512]``.
            key_value: BF16 key/value tensor ``[kv_sequence, kv_heads, 512]``.
            indices: Sparse key/value indices in ``int32``.
            attention_sink: Optional per-head attention sink.
            topk_length: Optional valid Top-K length per query token.

        Returns:
            Output, maximum logits, and log-sum-exp tensors.
        """
        return deepseek_flash_mla_sparse_prefill(
            query,
            key_value,
            indices,
            self.softmax_scale,
            value_head_dim=self.value_head_dim,
            attention_sink=attention_sink,
            topk_length=topk_length,
        )


class DeepseekV41SparseDecodeAttention(nn.Module):
    """Stateful DeepSeek V4.1 sparse-decode attention.

    FlashMLA caches scheduler metadata after the first invocation. Reuse this
    module only while batch/cache shapes and Top-K lengths stay unchanged, or
    call :meth:`reset_scheduler` before an input change.

    Scheduler metadata contains device tensors and is reset automatically when
    :meth:`torch.nn.Module.to` or another ``_apply``-based transform runs. A
    module instance is not safe for concurrent forwards with different input
    contracts.

    Args:
        value_head_dim: Value dimension. FlashMLA currently requires 512.
        softmax_scale: Optional scale applied to attention logits.
        enable_batch_invariant: Disable split-KV decoding for batch-invariant
            numerical results.
    """

    def __init__(
        self,
        value_head_dim: int = 512,
        softmax_scale: float | None = None,
        enable_batch_invariant: bool = False,
    ) -> None:
        """Initialize sparse-decode attention configuration."""
        super().__init__()
        _validate_value_head_dim(value_head_dim)
        self.value_head_dim = value_head_dim
        self.softmax_scale = softmax_scale
        self._scheduler_metadata: Any = None
        self._enable_batch_invariant = False
        self.enable_batch_invariant = enable_batch_invariant

    @property
    def enable_batch_invariant(self) -> bool:
        """Return whether batch-invariant FlashMLA decoding is enabled."""
        return self._enable_batch_invariant

    @enable_batch_invariant.setter
    def enable_batch_invariant(self, enabled: bool) -> None:
        """Update batch-invariant decoding and invalidate stale scheduling metadata."""
        if not isinstance(enabled, bool):
            raise ValueError(f"enable_batch_invariant must be bool, got {type(enabled).__name__}")
        if enabled != self._enable_batch_invariant:
            self.reset_scheduler()
        self._enable_batch_invariant = enabled

    def reset_scheduler(self) -> None:
        """Discard cached FlashMLA scheduling metadata."""
        self._scheduler_metadata = None

    def _apply(self, fn, recurse: bool = True):
        # Scheduler tensors belong to the device on which metadata was created.
        self.reset_scheduler()
        return super()._apply(fn, recurse=recurse)

    def _metadata(self) -> Any:
        if self._scheduler_metadata is None:
            metadata = deepseek_flash_mla_metadata()
            self._scheduler_metadata = metadata[0] if isinstance(metadata, tuple) else metadata
        return self._scheduler_metadata

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        indices: torch.Tensor,
        *,
        attention_sink: torch.Tensor | None = None,
        topk_length: torch.Tensor | None = None,
        extra_key_cache: torch.Tensor | None = None,
        extra_indices: torch.Tensor | None = None,
        extra_topk_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply FlashMLA sparse-decode attention.

        Args:
            query: BF16 query tensor ``[batch, query_sequence, heads, 512]``.
            key_cache: Quantized paged key/value cache.
            indices: Sparse cache indices in ``int32``.
            attention_sink: Optional per-head attention sink.
            topk_length: Optional valid Top-K length per batch item.
            extra_key_cache: Optional secondary quantized cache.
            extra_indices: Sparse indices into the secondary cache.
            extra_topk_length: Optional valid Top-K length for the secondary
                cache.

        Returns:
            Output and log-sum-exp tensors.
        """
        return deepseek_flash_mla_sparse_decode(
            query,
            key_cache,
            indices,
            self._metadata(),
            value_head_dim=self.value_head_dim,
            softmax_scale=self.softmax_scale,
            attention_sink=attention_sink,
            topk_length=topk_length,
            extra_key_cache=extra_key_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            enable_batch_invariant=self.enable_batch_invariant,
        )


def _validate_value_head_dim(value_head_dim: int) -> None:
    if value_head_dim != 512:
        raise ValueError(f"FlashMLA V4.1 requires value_head_dim=512, got {value_head_dim}")


__all__ = ["DeepseekV41SparseDecodeAttention", "DeepseekV41SparsePrefillAttention"]
