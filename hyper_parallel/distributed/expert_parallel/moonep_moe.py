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
"""A :class:`MoE` whose routed experts are moved by MoonEP.

The router, the auxiliary loss and the shared expert stay exactly as
``MoE`` has them; only the routed-experts step changes
(:meth:`MoE._routed_experts`).  Each rank holds its own slice of the experts and
MoonEP's plan decides which rows it computes: its own experts' rows always, plus
- when the router skews past one rank's capacity - the overflow of remote
experts, which this module recomputes from those experts' weights.

The remote weights come from a *differentiable* all-gather of the local shards,
so the redundant experts' gradients land on their owners.  That is the same
bytes MoonEP's ``prefetch_weight`` moves; replacing the gather with the MoonEP
primitive needs the expert parameters to live in MoonEP's symmetric memory, see
``docs/moonep_ep_integration.md`` §6.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_parallel.components.modules.moe import (
    GroupedExperts,
    MoE,
    _run_experts_grouped_mm_npu,
)
from hyper_parallel.distributed.expert_parallel.moonep_ep import MoonEPExpertParallel

__all__ = ["MoonEPMoE"]


class MoonEPMoE(MoE):
    """Expert-parallel MoE layer that dispatches tokens through MoonEP.

    Args:
        ep_group: EP process group (its size must divide ``num_experts``).
        dim: Hidden dimension.
        hidden_dim: Expert intermediate dimension.
        num_experts: Total routed experts across the EP group.
        top_k: Experts selected per token.
        score_before_experts: Mirror of :class:`MoE`'s flag; the routing weight
            is applied to the expert input (``True``) or output (``False``).
        router_kwargs: Extra arguments for :class:`TokenChoiceTopKRouter`.

    Note:
        ``self.experts`` holds *this rank's* experts only
        (``num_experts // ep_group.size()`` of them), so its parameters are the
        unit of EP sharding and of the optimizer state.
    """

    def __init__(
        self,
        ep_group: Any,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        score_before_experts: bool = True,
        router_kwargs: dict | None = None,
    ) -> None:
        """Build the router and this rank's expert shard, then bind the transport."""
        super().__init__(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k,
            score_before_experts=score_before_experts, router_kwargs=router_kwargs,
        )
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        if num_experts % self.ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must divide the EP group size ({self.ep_size})")
        self.local_experts = num_experts // self.ep_size
        # This rank's slice is what it owns; the router still spans all experts.
        self.experts = GroupedExperts(dim, hidden_dim, self.local_experts)
        self.transport = MoonEPExpertParallel(ep_group, num_experts, dim, rank=self.ep_rank)

    # -- expert shard --------------------------------------------------------

    def expert_table(self, weight: torch.Tensor) -> torch.Tensor:
        """Gather every rank's slice of *weight* into the global expert table.

        Differentiable: the gradient of a redundant segment flows back to the
        expert's owner through the gather's own backward.

        Args:
            weight: This rank's ``[local_experts, ...]`` expert weight.

        Returns:
            ``[num_experts, ...]`` weights in global expert order.
        """
        return torch.cat(
            torch.distributed.nn.functional.all_gather(weight, group=self.ep_group), dim=0)

    def segment_forward(self, segment: torch.Tensor, expert: int, tables: tuple,
                        weights: torch.Tensor) -> torch.Tensor:
        """Routed-expert math for one segment, mirroring :class:`MoE`.

        ``score_before_experts`` decides where the routing weight enters, exactly
        as in the default path: scaled onto the input, or onto the gated
        intermediate ``silu(w1 x) * w3 x`` before the ``w2`` projection.

        Args:
            segment: ``[rows, dim]`` rows delivered for *expert*.
            expert: Global expert id those rows belong to.
            tables: ``(w1, w2, w3)`` global expert tables, ``[E, ...]`` each.
            weights: ``[rows]`` routing weights delivered with the segment.

        Returns:
            ``[rows, dim]`` this segment's contribution.
        """
        w1, w2, w3 = tables
        rows = segment.float()
        if self.score_before_experts:
            rows = rows * weights.float().unsqueeze(-1)
            hidden = F.silu(rows @ w1[expert].t()) * (rows @ w3[expert].t())
        else:
            hidden = F.silu(rows @ w1[expert].t()) * (rows @ w3[expert].t())
        out = hidden @ w2[expert].t()
        # With ``score_before_experts=False`` the weight lands after the w2
        # projection, which is where the framework's grouped-mm variant puts it
        # too - the two MoonEP paths then agree with each other.
        return out if self.score_before_experts else out * weights.float().unsqueeze(-1)

    def _routed_experts(self, x_flat, top_scores, selected_experts, num_tokens, dim):
        """Move tokens with MoonEP and run local + redundant experts.

        See :meth:`MoE._routed_experts` for the contract.  Rows travel
        token-major (one row per ``(token, k)`` slot) with their expert ids, so
        the plan - not the row order - decides where each row lands.
        """
        del dim
        device = x_flat.device
        token_of_row = torch.arange(num_tokens, device=device).repeat_interleave(self.top_k)
        tables = (self.expert_table(self.experts.w1),
                  self.expert_table(self.experts.w2),
                  self.expert_table(self.experts.w3))
        out_flat = self.transport.forward(
            x_flat[token_of_row].to(torch.bfloat16),
            selected_experts.reshape(-1).to(torch.int32),
            top_scores.reshape(-1).to(torch.float32),
            expert_forward=lambda segment, expert, weights: self.segment_forward(
                segment, expert, tables, weights),
            expert_forward_many=self._grouped_segments,
            output_dtype=torch.float32,
            requires_route_weights=True)
        # The gathered tables' backward (a reduce_scatter) is collective and
        # must fire on every rank.  The grouped path (_grouped_segments) reads
        # only the local weight shard, leaving the gather out of that rank's
        # graph and deadlocking the step -- so anchor the output to a zero
        # read of each table, the same trick as moonep_forward's
        # _tables_anchor.
        anchor = sum(t.reshape(-1)[0].float() * 0.0 for t in tables)
        out_flat = out_flat + anchor
        # Combine accumulates in fp32 by design, but the block output must keep
        # the residual stream's dtype (the default MoE._routed_experts follows
        # x).  Same pattern as moonep_forward's production seam.
        return torch.zeros(num_tokens, x_flat.shape[1], dtype=x_flat.dtype,
                           device=device).index_add(
                               0, token_of_row, out_flat.to(x_flat.dtype))

    def _grouped_segments(self, rows_local: torch.Tensor, local_counts: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
        """Run every segment with one grouped SwiGLU GEMM.

        Only called when the layout is this rank's own experts in ascending
        order (see the transport's check), so ``local_counts`` describes the
        groups of this rank's weight table exactly.

        Args:
            rows_local: ``[received_rows, dim]`` bf16 rows, local-expert order.
            local_counts: ``[local_experts]`` rows per local expert (zeros kept).
            weights: ``[received_rows]`` routing weights of those rows.

        Returns:
            ``[received_rows, dim]`` expert output with the routing weight
            applied per ``score_before_experts``.
        """
        rows = rows_local.float()
        scores = weights.float() if self.score_before_experts else None
        if self.score_before_experts:
            rows = rows * scores.unsqueeze(-1)
        output = _run_experts_grouped_mm_npu(
            self.experts.w1, self.experts.w2, self.experts.w3, rows,
            local_counts.to(rows_local.device), scores=None if self.score_before_experts else scores)
        return output
