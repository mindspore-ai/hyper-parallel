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
"""Training objectives around the official Xing4 router, experts and MLPs."""
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.multicore.modules._precision import preserve_fp32_tensors
from hyper_parallel.core.utils.communication import differentiable_all_reduce
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import (
    Xing4_0MLP, Xing4_0MoE, Xing4_0TopkRouter,
)


class Xing4Router(Xing4_0TopkRouter):
    """HF learned selection with the recipe's training state and objectives."""

    fp32_buffer_names = ("e_score_correction_bias",)

    def __init__(self, config: Any, source: nn.Module) -> None:
        """Retain the HF weights and register all resumable routing state."""
        super().__init__(config)
        self.weight = source.weight
        self.e_score_correction_bias = source.e_score_correction_bias
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.aux_loss_cp_group = None
        self.register_buffer("_token_counts", torch.zeros(self.num_experts, dtype=torch.int64))
        self.input_jitter_eps = config.moe_input_jitter_eps
        self.last_aux_loss = None
        self.enable_expert_bias = config.moe_router_enable_expert_bias
        self.expert_bias_update_rate = config.moe_router_bias_update_rate

    def _apply(self, fn: Callable, recurse: bool = True) -> "Xing4Router":
        """Keep load-balancing bias updates in FP32 during model dtype conversion."""
        return super()._apply(preserve_fp32_tensors(self, fn, self.fp32_buffer_names), recurse=recurse)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route real tokens once; safe-token padding consumes neither jitter nor counts."""
        real_hidden = hidden_states[:, self.padding:]
        flat = real_hidden.reshape(-1, hidden_states.shape[-1]).float()
        if self.training and self.input_jitter_eps:
            flat = flat * (1.0 + (2.0 * torch.rand_like(flat) - 1.0) * self.input_jitter_eps)
        logits, weights, indices = super().forward(flat)
        scores = logits.sigmoid()
        weights = self._attach_balance_loss(weights, scores, batch_size=hidden_states.shape[0])
        if self.training and self.enable_expert_bias and not is_recomputing():
            counts = torch.zeros_like(self._token_counts).scatter_add_(
                0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=torch.int64))
            self._token_counts.add_(counts)
        if self.padding:
            batch = hidden_states.shape[0]
            padding = torch.arange(self.padding * self.top_k, device=indices.device)
            padding = (padding.view(1, self.padding, self.top_k) % self.num_experts).expand(batch, -1, -1)
            indices = torch.cat((padding, indices.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
            weights = torch.cat((weights.new_zeros(batch, self.padding, self.top_k),
                                 weights.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
        return logits, weights.to(hidden_states.dtype), indices

    def _attach_balance_loss(self, weights: torch.Tensor, scores: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Inject ``coeff * E * mean_b(sum_i(f_bi * P_bi))`` through the routing weights.

        ``f_bi`` counts unbiased top-k affinities per sequence, independently of the expert-load bias. ``P_bi`` averages
        the normalized affinities within each sequence. The sequence objectives are averaged over
        the batch.

        Args:
            weights: Routing weights of the real tokens.
            scores: Router affinities of every expert.
            batch_size: Number of independent sequences before flattening the tokens.
        """
        selected_indices = scores.topk(self.top_k, dim=-1, sorted=False).indices
        selected = selected_indices.reshape(batch_size, -1)
        counts = scores.new_zeros(batch_size, self.num_experts).scatter_add_(
            1, selected, torch.ones_like(selected, dtype=scores.dtype))
        if getattr(self, 'aux_loss_cp_group', None) is None:
            load = counts / selected.shape[1]
            affinity = (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).reshape(
                batch_size, -1, self.num_experts).mean(dim=1)
        else:
            normalized = (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).reshape(
                batch_size, -1, self.num_experts)
            group = self.aux_loss_cp_group
            with torch.no_grad():
                counts = counts.detach()
                dist.all_reduce(counts, group=group)
                tokens = counts.new_tensor(normalized.shape[1])
                dist.all_reduce(tokens, group=group)
            load = counts / (tokens * self.top_k)
            affinity = differentiable_all_reduce(normalized.sum(dim=1), "sum", group) / tokens
        loss = (affinity * load).sum(dim=-1).mean() * self.num_experts
        self.last_aux_loss = loss.detach()
        return aux_loss_auto_scale(weights, loss * self.aux_loss_coeff)


    @property
    def tokens_per_expert(self) -> torch.Tensor:
        """Expose the persistent counts to the shared MoE monitor."""
        return self._token_counts

    @torch.no_grad()
    def update_expert_bias(self, lr: float, num_recomputations: int = 1) -> None:
        """Apply one load-based update after the optimizer step."""
        del num_recomputations
        counts = self._token_counts
        self.e_score_correction_bias.add_((counts.sum() - counts * self.num_experts).sign(), alpha=lr)
        counts.zero_()


class Xing4MLP(Xing4_0MLP):
    """Retain HF projections with an optional training activation policy."""

    def __init__(self, source: nn.Module) -> None:
        """Adopt the HF parameters without creating a second weight owner."""
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.config = source.config
        self.hidden_size = source.hidden_size
        self.intermediate_size = source.intermediate_size
        self.gate_proj, self.up_proj, self.down_proj = source.gate_proj, source.up_proj, source.down_proj
        # Only the optional clipped/compute-dtype policy needs the extended SwiGLU API.
        from hyper_parallel.components.functional.activation_swiglu import (  # pylint: disable=import-outside-toplevel
            swiglu_from_projections,
        )
        self.act_fn = source.act_fn
        self._swiglu = swiglu_from_projections
        self.limit = getattr(self.config, "swiglu_limit", None)
        dtype_name = getattr(self.config, "swiglu_compute_dtype", None)
        self.activation_compute_dtype = getattr(torch, dtype_name) if dtype_name is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured shared SwiGLU policy to the HF projections."""
        activation = self._swiglu(
            self.gate_proj(x), self.up_proj(x), clamp_limit=self.limit,
            compute_dtype=self.activation_compute_dtype,
            use_fused=bool(getattr(self.config, "use_fused_swiglu", True)))
        return self.down_proj(activation)


def adapt_mlp(mlp: nn.Module) -> nn.Module:
    """Apply the training activation policy when requested."""
    config = mlp.config
    if getattr(config, "swiglu_limit", None) is None and getattr(config, "swiglu_compute_dtype", None) is None:
        return mlp
    return Xing4MLP(mlp)


class Xing4MoE(Xing4_0MoE):
    """Reuse the HF MoE facade and experts with explicit training extensions."""

    def __init__(self, config: Any, source: nn.Module) -> None:
        """Retain HF experts and add the training router and EP execution boundary."""
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.config = config
        self.experts = nn.ModuleList(adapt_mlp(expert) for expert in source.experts)
        self.gate = Xing4Router(config, source.gate)
        self.shared_experts = adapt_mlp(source.shared_experts)
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.ep_executor = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reuse HF expert computation, with optional padding and EP dispatch."""
        padded = hidden_states
        if self.padding:
            padded = torch.cat((padded.new_zeros(padded.shape[0], self.padding, padded.shape[-1]), padded), dim=1)
        if self.ep_executor is not None:
            routed = self.ep_executor(self, padded)
            return routed[:, self.padding:] + self.shared_experts(hidden_states)
        return super().forward(padded)[:, self.padding:]

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor,
            topk_weights: torch.Tensor) -> torch.Tensor:
        """Select the HF per-expert or declared grouped-expert implementation."""
        if isinstance(self.experts, nn.ModuleList):
            return super().moe(hidden_states, topk_indices, topk_weights)
        return self.experts(hidden_states, topk_indices, topk_weights)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the same once-only router contract to the expert-parallel executor."""
        _, weights, indices = self.gate(hidden)
        return indices, weights
