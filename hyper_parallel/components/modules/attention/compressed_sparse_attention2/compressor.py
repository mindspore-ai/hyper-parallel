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
"""Learned non-overlapping CSA2 compression with an explicit block exchange stage."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.rms_norm_family import NativeRMSNormModule, RMSNorm


class CSA2Compressor(nn.Module):
    """Non-overlapping V4.1 compressed-KV producer."""

    def __init__(self, config: Any, compress_ratio: int, *, apply_rope: Callable) -> None:
        """Create the learned KV pooling projections."""
        super().__init__()
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        self.compress_ratio = compress_ratio
        self.apply_rope = apply_rope
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        if compress_ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        # This is a source subtree until the enclosing CSA2 family is selected.
        norm = NativeRMSNormModule(
            config=SimpleNamespace(hidden_size=config.head_dim, eps=config.rms_norm_eps),
            cast_order="before_weight",
        )
        self.norm = RMSNorm(module=norm)

    @staticmethod
    def pool(projected: torch.Tensor, plan: Any) -> torch.Tensor:
        """Pool complete document blocks in an already prepared projected buffer."""
        rows = plan.indices(projected.device)[0]
        blocks = projected.index_select(1, rows).reshape(
            projected.shape[0], rows.numel() // plan.ratio, plan.ratio, projected.shape[-1],
        )
        values, scores = blocks.chunk(2, dim=-1)
        return (values * scores.float().softmax(dim=2).to(values.dtype)).sum(dim=2)

    def exchange_blocks(self, projected: torch.Tensor, block_plan: Any) -> torch.Tensor:
        """Keep complete local blocks in place; parallel members complete boundary blocks."""
        del block_plan
        return projected

    def forward(
            self,
            hidden_states: torch.Tensor,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            block_plan: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unrotated and RoPE-rotated compressed KV tensors.

        Args:
            hidden_states: Decoder hidden rows to compress.
            compress_position_embeddings: Rotary coefficients for compressed keys.
            block_plan: Optional document-local grouping and cross-rank boundary plan.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        if self.compress_ratio == 1:
            latent = self.norm(self.wkv(hidden_states))
            cos, sin = compress_position_embeddings
            rotated = self.apply_rope(latent.unsqueeze(1), cos, sin).squeeze(1)
            return latent, rotated

        if block_plan is not None:
            projected = torch.cat((self.wkv(hidden_states), self.wgate(hidden_states)), dim=-1)
            projected = self.exchange_blocks(projected, block_plan)
            latent = self.norm(self.pool(projected, block_plan))
            cos, sin = compress_position_embeddings
            return latent, self.apply_rope(
                latent.unsqueeze(1), cos, sin,
            ).squeeze(1)

        usable = sequence_length - sequence_length % self.compress_ratio
        key_value = self.wkv(hidden_states[:, :usable])
        gate = self.wgate(hidden_states[:, :usable])
        key_value = key_value.view(batch_size, -1, self.compress_ratio, key_value.shape[-1])
        gate = gate.view(batch_size, -1, self.compress_ratio, gate.shape[-1])
        latent = self.norm((key_value * gate.float().softmax(dim=2).to(key_value.dtype)).sum(dim=2))
        cos, sin = compress_position_embeddings
        cos = cos[:, :usable:self.compress_ratio]
        sin = sin[:, :usable:self.compress_ratio]
        rotated = self.apply_rope(latent.unsqueeze(1), cos, sin).squeeze(1)
        return latent, rotated
