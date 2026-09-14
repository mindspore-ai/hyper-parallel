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
"""Reusable DeepSeek Engram hash, sparse lookup, and fusion modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.models.replacement import module_replacement


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
        layer_ids = tuple(int(value) for value in assets["layer_ids"])
        if layer_id not in layer_ids:
            raise ValueError(f"Engram layer {layer_id} is absent from assets layer_ids={layer_ids}")
        layer_index = layer_ids.index(layer_id)
        self.layer_id = layer_id
        self.max_ngram_size = int(assets["max_ngram_size"])
        self.num_heads = int(assets["num_heads"])
        token_map = torch.tensor(assets["token_map"], dtype=torch.long)
        self.pad_id = int(token_map[int(assets["pad_token_id"])])
        primes = torch.tensor(assets["primes"][layer_index], dtype=torch.long)
        if tuple(primes.shape) != (self.max_ngram_size - 1, self.num_heads):
            raise ValueError(
                "Engram prime layout must be [max_ngram_size - 1, num_heads], "
                f"got {tuple(primes.shape)}"
            )
        flattened_primes = primes.flatten()
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long), flattened_primes.cumsum(0)[:-1]]
        ).view_as(primes)
        multipliers = torch.tensor(assets["multipliers"][layer_index], dtype=torch.long)
        if multipliers.numel() != self.max_ngram_size:
            raise ValueError("Engram multiplier count must equal max_ngram_size")
        self.logical_num_embeddings = int(flattened_primes.sum().item())
        self._initial_buffers = {
            "token_map": token_map,
            "primes": primes,
            "offsets": offsets,
            "multipliers": multipliers,
        }
        for name, value in self._initial_buffers.items():
            self.register_buffer(name, value.clone(), persistent=False)
        self._hp_reset_after_materialization = True

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Restore hash metadata after meta-device materialization."""
        for name, value in self._initial_buffers.items():
            getattr(self, name).copy_(value.to(getattr(self, name).device))

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return global flat-table rows shaped ``[batch, seq, hash_columns]``."""
        if input_ids.ndim != 2:
            raise ValueError(f"Engram expects input_ids [B,S], got {tuple(input_ids.shape)}")
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
        for shift in range(self.max_ngram_size):
            source_positions = (positions - shift).clamp_min(0)
            source = compressed.gather(1, source_positions)
            blocked = positions < shift
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


def _all_gather_sequence(value: torch.Tensor, group: Any, group_size: int) -> torch.Tensor:
    """Gather equal-size sequence chunks and concatenate along dimension one."""
    gathered = [torch.empty_like(value) for _ in range(group_size)]
    dist.all_gather(gathered, value.contiguous(), group=group)
    return torch.cat(gathered, dim=1)


@module_replacement
class EngramModule(nn.Module):
    """Structure-preserving Engram replacement with sparse EP row lookup."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Reuse a source Engram module's parameters and hash mapping."""
        super().__init__()
        del module_fqn, context
        required = ("hash_mapping", "embed", "wkv", "q_weight", "k_weight")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"Engram source is missing required attributes: {missing}")
        self.hash_mapping = module.hash_mapping
        self.embed = module.embed
        self.wkv = module.wkv
        self.q_weight = module.q_weight
        self.k_weight = module.k_weight
        self.layer_id = int(module.layer_id)
        self.hidden_size = int(module.hidden_size)
        self.hc_mult = int(module.hc_mult)
        self.eps = float(module.eps)
        self.clamp_value = float(module.clamp_value)
        self.logical_num_embeddings = int(module.logical_num_embeddings)
        self.padded_num_embeddings = int(module.padded_num_embeddings)
        self.train(module.training)

    def _fuse(self, hidden_states: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the exact V4.1 WKV projection and signed-root gate."""
        key_value = self.wkv(embeddings.flatten(start_dim=-2))
        key, value = key_value.split(
            [self.hc_mult * self.hidden_size, self.hidden_size], dim=-1
        )
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        hidden_fp32 = hidden_states.float()
        weight = self.q_weight.float() * self.k_weight.float()
        reciprocal_std = torch.rsqrt(hidden_fp32.square().mean(-1) + self.eps)
        reciprocal_std = reciprocal_std * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden_fp32 * weight * key).sum(-1)
        dot = dot * reciprocal_std * self.hidden_size**-0.5
        root = dot.abs().clamp_min(self.clamp_value).sqrt()
        signed_root = torch.where(dot < 0, -root, root)
        gate = torch.sigmoid(signed_root)
        return (
            hidden_fp32 + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        ).to(hidden_states.dtype)

    @staticmethod
    def _sparse_lookup(
        hash_ids: torch.Tensor,
        weight: torch.Tensor,
        *,
        ep_group: Any,
        ep_rank: int,
        ep_size: int,
        padded_num_embeddings: int,
    ) -> torch.Tensor:
        """Route requested global rows to their EP owners and restore order."""
        if padded_num_embeddings % ep_size:
            raise ValueError(
                f"Engram padded rows {padded_num_embeddings} are not divisible by ep_size={ep_size}"
            )
        num_local_rows = padded_num_embeddings // ep_size
        row_start = ep_rank * num_local_rows
        flat_ids = hash_ids.reshape(-1)
        owner = torch.div(flat_ids, num_local_rows, rounding_mode="floor")
        if owner.numel() and int(owner.max()) >= ep_size:
            raise IndexError("Engram hash requested a row outside the padded table")
        local_counts = torch.bincount(owner, minlength=ep_size).to(torch.long)
        gathered_counts = [torch.empty_like(local_counts) for _ in range(ep_size)]
        dist.all_gather(gathered_counts, local_counts, group=ep_group)
        all_counts = torch.stack(gathered_counts)
        send_counts = local_counts.cpu().tolist()
        recv_counts = all_counts[:, ep_rank].cpu().tolist()

        # Ascend ArgSort executes integer inputs on AiCPU. The owner range is
        # bounded by ep_size, so float32 preserves it exactly and stays on AiCore.
        sort_indices = torch.argsort(owner.float(), stable=True)
        sorted_ids = flat_ids.index_select(0, sort_indices)
        received_ids = ep_all_to_all(sorted_ids, send_counts, recv_counts, ep_group)
        local_ids = received_ids - row_start
        if local_ids.numel() and (int(local_ids.min()) < 0 or int(local_ids.max()) >= weight.shape[0]):
            raise IndexError("Engram EP routing delivered a row to the wrong owner")
        values = F.embedding(local_ids, weight)
        returned = ep_all_to_all(values, recv_counts, send_counts, ep_group)
        restored = torch.zeros_like(returned)
        restored.scatter_add_(
            0,
            sort_indices.unsqueeze(1).expand(-1, returned.shape[1]),
            returned,
        )
        return restored.view(*hash_ids.shape, weight.shape[1])

    def _aligned_hash_ids(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None,
        token_mask: torch.Tensor | None = None,
        *,
        cp_group: Any = None,
        cp_rank: int = 0,
        cp_size: int = 1,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> torch.Tensor:
        """Hash the local sequence plus the left halo required by n-grams."""
        cp_local_ids = input_ids
        if cp_size > 1:
            input_ids = _all_gather_sequence(input_ids, cp_group, cp_size)
            if segment_starts is not None:
                segment_starts = _all_gather_sequence(segment_starts, cp_group, cp_size)
            if token_mask is not None:
                token_mask = _all_gather_sequence(token_mask, cp_group, cp_size)
        local_sequence = hidden_states.shape[1]
        cp_sequence = cp_local_ids.shape[1]
        if cp_sequence == local_sequence:
            sequence_parallel_parts = 1
            local_tp_rank = 0
        elif cp_sequence % local_sequence == 0 and cp_sequence // local_sequence == tp_size:
            sequence_parallel_parts = tp_size
            local_tp_rank = tp_rank
        else:
            raise ValueError(
                "Engram input_ids length is incompatible with the local hidden sequence: "
                f"ids={cp_sequence}, hidden={local_sequence}, tp_size={tp_size}"
            )
        del sequence_parallel_parts
        start = cp_rank * cp_sequence + local_tp_rank * local_sequence
        end = start + local_sequence
        context_start = max(0, start - self.hash_mapping.max_ngram_size + 1)
        ids_window = input_ids[:, context_start:end]
        starts_window = None if segment_starts is None else segment_starts[:, context_start:end]
        mask_window = None if token_mask is None else token_mask[:, context_start:end]
        hashes = self.hash_mapping(ids_window, starts_window, mask_window)
        target_offset = start - context_start
        return hashes[:, target_offset:target_offset + local_sequence]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run replicated single-card/EP1 Engram."""
        hash_ids = self._aligned_hash_ids(hidden_states, input_ids, segment_starts, token_mask)
        fused = self._fuse(hidden_states, self.embed(hash_ids))
        if token_mask is None:
            return fused
        return torch.where(token_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1), fused, hidden_states)

    def parallel_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        *,
        ep_group: Any,
        ep_rank: int,
        ep_size: int,
        cp_group: Any = None,
        cp_rank: int = 0,
        cp_size: int = 1,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> torch.Tensor:
        """Run CP/TP sequence alignment and sparse EP lookup."""
        hash_ids = self._aligned_hash_ids(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask,
            cp_group=cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        embeddings = self._sparse_lookup(
            hash_ids,
            self.embed.weight,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            padded_num_embeddings=self.padded_num_embeddings,
        )
        fused = self._fuse(hidden_states, embeddings)
        if token_mask is None:
            return fused
        return torch.where(token_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1), fused, hidden_states)


__all__ = ["EngramModule", "NgramHashMapping"]
