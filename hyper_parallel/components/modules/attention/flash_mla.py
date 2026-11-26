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
"""Forward-only sparse attention modules with caller-owned FlashMLA scheduler state."""

from __future__ import annotations

import torch
from torch import nn

from hyper_parallel.components.ops import flash_mla_sparse_decode, flash_mla_sparse_prefill


class FlashMLASparsePrefill(nn.Module):
    """Sparse prefill through the configured attention backend, without training autograd."""

    def __init__(self, softmax_scale: float, value_head_dim: int = 512) -> None:
        """Configure sparse prefill scale and value width."""
        super().__init__()
        self.softmax_scale, self.value_head_dim = softmax_scale, value_head_dim

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, indices: torch.Tensor, *,
                attention_sink: torch.Tensor | None = None,
                topk_length: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return output, max logits and LSE; the sink changes output normalization only."""
        return flash_mla_sparse_prefill(query, key_value, indices, self.softmax_scale,
                                        value_head_dim=self.value_head_dim, attention_sink=attention_sink,
                                        topk_length=topk_length)


class FlashMLASparseDecode(nn.Module):
    """Cache scheduler metadata only while its shape and length assumptions remain valid.

    Length-bearing calls always get fresh native metadata: checking tensor
    identity would miss in-place updates, and reading NPU lengths on the host
    would introduce a synchronization point.
    """

    def __init__(self, softmax_scale: float | None = None, value_head_dim: int = 512,
                 enable_batch_invariant: bool = False) -> None:
        """Configure decode normalization and initialize empty scheduler state."""
        super().__init__()
        self.softmax_scale, self.value_head_dim = softmax_scale, value_head_dim
        self._state: dict = {}
        self._signature = None
        self.enable_batch_invariant = enable_batch_invariant

    def reset_scheduler(self) -> None:
        """Discard transient metadata without adding it to checkpoint state."""
        self._state, self._signature = {}, None

    def _apply(self, fn, recurse: bool = True):
        self.reset_scheduler()
        return super()._apply(fn, recurse=recurse)

    def forward(self, query: torch.Tensor, key_cache: torch.Tensor, indices: torch.Tensor, *,
                attention_sink: torch.Tensor | None = None, topk_length: torch.Tensor | None = None,
                extra_key_cache: torch.Tensor | None = None, extra_indices: torch.Tensor | None = None,
                extra_topk_length: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode using physical page-slot IDs, with -1 marking unused indices."""
        signature = (query.device, query.dtype, tuple(query.shape), tuple(key_cache.shape), key_cache.dtype,
                     tuple(indices.shape), None if extra_key_cache is None else tuple(extra_key_cache.shape),
                     None if extra_indices is None else tuple(extra_indices.shape), self.enable_batch_invariant)
        if signature != self._signature or topk_length is not None or extra_topk_length is not None:
            self.reset_scheduler()
            self._signature = signature
        return flash_mla_sparse_decode(query, key_cache, indices, self._state,
                                       value_head_dim=self.value_head_dim, softmax_scale=self.softmax_scale,
                                       attention_sink=attention_sink, topk_length=topk_length,
                                       extra_key_cache=extra_key_cache, extra_indices=extra_indices,
                                       extra_topk_length=extra_topk_length,
                                       enable_batch_invariant=self.enable_batch_invariant)
