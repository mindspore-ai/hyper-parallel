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
"""Deterministic n-gram hash assets and rebuildable metadata."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn
from hyper_parallel.models.materialization import register_rebuildable_buffer


class NgramHashMapping(nn.Module):
    """Map token IDs to one Engram layer's disjoint n-gram hash rows."""

    DEAD_TOKEN = -1

    def __init__(self, assets: Mapping[str, Any], layer_id: int) -> None:
        """Create deterministic, non-persistent hash metadata.

        Args:
            assets: Prepared tokenizer map and per-layer hash layout.
            layer_id: Decoder layer owning this Engram table.
        """
        super().__init__()
        target_device = torch.empty(0).device
        layer_ids = tuple(int(value) for value in assets["layer_ids"])
        if layer_id not in layer_ids:
            raise ValueError(f"Engram layer {layer_id} is absent from assets layer_ids={layer_ids}")
        layer_index = layer_ids.index(layer_id)
        self.layer_id = layer_id
        self.max_ngram_size = int(assets["max_ngram_size"])
        self.num_heads = int(assets["num_heads"])
        token_map = torch.tensor(assets["token_map"], dtype=torch.long, device="cpu")
        self.pad_id = int(token_map[int(assets["pad_token_id"])])
        primes = torch.tensor(assets["primes"][layer_index], dtype=torch.long, device="cpu")
        if tuple(primes.shape) != (self.max_ngram_size - 1, self.num_heads):
            raise ValueError(
                "Engram prime layout must be [max_ngram_size - 1, num_heads], "
                f"got {tuple(primes.shape)}"
            )
        flattened_primes = primes.flatten()
        if self.max_ngram_size < 2 or self.num_heads <= 0 or bool((primes <= 0).any()):
            raise ValueError("Engram requires ngram size >=2, positive heads and positive hash moduli")
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device="cpu"), flattened_primes.cumsum(0)[:-1]]
        ).view_as(primes)
        multipliers = torch.tensor(assets["multipliers"][layer_index], dtype=torch.long, device="cpu")
        if multipliers.numel() != self.max_ngram_size:
            raise ValueError("Engram multiplier count must equal max_ngram_size")
        self.logical_num_embeddings = int(flattened_primes.sum().item())
        initial_buffers = {
            "token_map": token_map,
            "primes": primes,
            "offsets": offsets,
            "multipliers": multipliers,
        }
        for name, value in initial_buffers.items():
            register_rebuildable_buffer(self, name, value=value, persistent=False)
            if target_device.type != "cpu":
                self.register_buffer(name, getattr(self, name).to(target_device), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return global flat-table rows shaped ``[batch, seq, hash_columns]``.

        Args:
            input_ids: Token IDs for hash lookup.
            segment_starts: Optional packed sequence boundaries.
            token_mask: Optional mask excluding image tokens.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Engram expects input_ids [B,S], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] == 0:
            return input_ids.new_empty((input_ids.shape[0], 0, (self.max_ngram_size - 1) * self.num_heads),
                                       dtype=torch.long)
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            if token_mask.shape != input_ids.shape:
                raise ValueError("Engram token_mask must have the same shape as input_ids")
            compressed = torch.where(
                token_mask.to(torch.bool),
                compressed,
                compressed.new_full((), self.DEAD_TOKEN),
            )
        batch_size, sequence_length = compressed.shape
        positions = torch.arange(sequence_length, device=input_ids.device).expand(batch_size, -1)
        segment_ids = None
        if segment_starts is not None:
            if segment_starts.shape != input_ids.shape:
                raise ValueError("Engram segment_starts must have the same shape as input_ids")
            starts = segment_starts.to(torch.bool).clone()
            starts[:, 0] = True
            segment_ids = starts.to(torch.long).cumsum(dim=1)

        tokens = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source_positions = (positions - shift).clamp_min(0)
            source = compressed.gather(1, source_positions)
            blocked = blocked | (positions < shift)
            if segment_ids is not None:
                source_segments = segment_ids.gather(1, source_positions)
                blocked = blocked | (source_segments != segment_ids)
            blocked = blocked | (source == self.DEAD_TOKEN)
            tokens.append(torch.where(blocked, self.pad_id, source))

        products = torch.stack(tokens, dim=-1) * self.multipliers
        rolling = products[..., 0]
        hashes = []
        for ngram_index in range(self.max_ngram_size - 1):
            rolling = torch.bitwise_xor(rolling, products[..., ngram_index + 1])
            row = rolling.unsqueeze(-1) % self.primes[ngram_index]
            hashes.append(row + self.offsets[ngram_index])
        return torch.cat(hashes, dim=-1)
