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
"""MLA sequence/head exchanges using the existing differentiable collectives."""

# CP uses the native Torch runtime, following the existing CP wrappers.
# pylint: disable=forbidden-backend-import

from typing import Any, Literal

import torch

from hyper_parallel.core.utils.communication import differentiable_all_gather_concat
from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head


MLACPStrategy = Literal["expanded_ulysses", "latent_kv_head"]


class MLAContextParallel:
    """Redistribute MLA activations without owning model projections or kernels."""

    def __init__(self, cp_mesh: Any, local_heads: int, strategy: MLACPStrategy) -> None:
        """Validate the TP-local head count against the requested CP degree.

        Args:
            cp_mesh: One-dimensional context-parallel mesh.
            local_heads: Number of heads in this TP parameter shard.
            strategy: Expanded QKV exchange or shared latent KV gathering.
        """
        self.mesh = cp_mesh
        self.degree = 1 if cp_mesh is None else cp_mesh.size()
        if getattr(cp_mesh, "ndim", 1) != 1 or self.degree < 1:
            raise ValueError("MLA CP requires a one-dimensional nonempty mesh")
        if local_heads < 1 or local_heads % self.degree:
            raise ValueError("MLA heads must be divisible by TP degree times CP degree")
        if strategy not in ("expanded_ulysses", "latent_kv_head"):
            raise ValueError(f"Unknown MLA CP strategy: {strategy}")
        self.strategy = strategy
        self.local_heads = local_heads
        self.compute_heads = local_heads // self.degree
        self.rank = 0 if cp_mesh is None else cp_mesh.get_local_rank()
        self.head_range = (self.rank * self.compute_heads, (self.rank + 1) * self.compute_heads)

    def expanded(self, query: torch.Tensor, key: torch.Tensor,
                 value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exchange unequal-width BNSD QKV in a single packed AllToAll.

        Args:
            query: Local-token query heads.
            key: Local-token key heads.
            value: Local-token value heads.
        """
        if self.degree == 1:
            return query, key, value
        widths = (query.shape[-1], key.shape[-1], value.shape[-1])
        payload = torch.cat(tuple(tensor.transpose(1, 2) for tensor in (query, key, value)), dim=-1)
        payload = ulysses_seq_to_head(payload, 1, 2, self.mesh)
        return tuple(part.transpose(1, 2) for part in payload.split(widths, dim=-1))

    def latent(self, query: torch.Tensor, latent: torch.Tensor,
               key_rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exchange Q heads and gather shared KV states before head expansion.

        Args:
            query: BNSD query with all TP-local heads and local tokens.
            latent: BSR normalized shared KV latent.
            key_rope: BSR rotated shared RoPE key.
        """
        if self.degree == 1:
            return query, latent, key_rope
        query = ulysses_seq_to_head(query, 2, 1, self.mesh)
        payload = torch.cat((latent, key_rope), dim=-1)
        payload = differentiable_all_gather_concat(payload, self.mesh.get_group(), self.degree, 1)
        latent, key_rope = payload.split((latent.shape[-1], key_rope.shape[-1]), dim=-1)
        return query, latent.contiguous(), key_rope

    def restore(self, output: torch.Tensor) -> torch.Tensor:
        """Return BSND attention output to local tokens and all TP-local heads.

        Args:
            output: Full-sequence output with this CP rank's compute heads.
        """
        if self.degree == 1:
            return output
        return ulysses_head_to_seq(output, 1, 2, self.mesh)
