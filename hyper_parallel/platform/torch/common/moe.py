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
"""PyTorch MoE building blocks: router, experts and orchestrator.

Provides :class:`FeedForward`, :class:`GroupedExperts`,
:class:`TokenChoiceTopKRouter`, and the top-level :class:`MoE` orchestrator.
Load-balancing utilities (expert bias update and auxiliary loss) are also
included.

All modules compose naturally with EP / TP parallel strategies via DTensor.
Distributed collectives are handled by
:mod:`hyper_parallel.core.expert_parallel.expert_parallel`; this module
contains only single-device computation.
"""
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from hyper_parallel.core.dtensor.dtensor import DTensor

__all__ = [
    "FeedForward",
    "GroupedExperts",
    "TokenChoiceTopKRouter",
    "MoE",
    "update_expert_bias",
]


# ---------------------------------------------------------------------------
# Grouped expert computation kernels
# ---------------------------------------------------------------------------

def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run per-expert SwiGLU via a sequential loop (reference path).

    Args:
        w1: Shape ``[num_experts, hidden_dim, dim]``.
        w2: Shape ``[num_experts, dim, hidden_dim]``.
        w3: Shape ``[num_experts, hidden_dim, dim]``.
        x: Routed tokens in expert-major order, shape
            ``[total_routed_tokens, dim]``.
        num_tokens_per_expert: 1-D integer tensor of length ``num_experts``.

    Returns:
        Expert output of shape ``[total_routed_tokens, dim]``.
    """
    # Use ``torch.cat`` instead of ``zeros_like + in-place slice assign``.
    # On some backends (notably torch_npu) in-place slice assignment onto a
    # ``requires_grad=False`` leaf tensor does not reliably upgrade it to a
    # non-leaf with a ``grad_fn`` — the forward result may end up with
    # ``grad_fn=None`` and downstream ``backward()`` fails with
    # "element 0 of tensors does not require grad and does not have a grad_fn".
    #
    # Drain ``num_tokens_per_expert`` to host **once** via ``.tolist()``
    # rather than calling ``int(n)`` per loop iteration — a single D2H
    # copy instead of ``num_local_experts`` separate ones.  Per-iter
    # ``.item()`` would stall the host between expert kernels and shrink
    # the dual-pipe overlap window.
    counts_list = num_tokens_per_expert.tolist()
    parts = []
    offset = 0
    for e, n in enumerate(counts_list):
        if n == 0:
            continue
        x_e = x[offset:offset + n]
        h = F.silu(x_e @ w1[e].T) * (x_e @ w3[e].T)
        parts.append(h @ w2[e].T)
        offset += n
    if not parts:
        # No routed tokens: return a grad-connected zero (not ``zeros_like``).
        return x * 0.0
    return torch.cat(parts, dim=0)


def _run_experts_grouped_mm_gpu(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Fused grouped matmul path for NVIDIA GPU using ``torch._grouped_mm``.

    Args:
        w1: Shape ``[num_experts, hidden_dim, dim]``.
        w2: Shape ``[num_experts, dim, hidden_dim]``.
        w3: Shape ``[num_experts, hidden_dim, dim]``.
        x: Shape ``[total_routed_tokens, dim]``.
        num_tokens_per_expert: 1-D integer tensor of length ``num_experts``.

    Returns:
        Expert output of shape ``[total_routed_tokens, dim]``.
    """
    # offs: cumulative split offsets (int32) for torch._grouped_mm.
    offs = torch.cumsum(num_tokens_per_expert[:-1], dim=0).to(torch.int32)
    # w1/w3 stored as [num_experts, hidden_dim, dim]; grouped_mm expects
    # [num_experts, dim, hidden_dim], so transpose the inner two dims.
    w1_t = w1.transpose(1, 2).contiguous()  # [num_experts, dim, hidden_dim]
    w3_t = w3.transpose(1, 2).contiguous()
    w2_t = w2.transpose(1, 2).contiguous()  # [num_experts, hidden_dim, dim]
    h1 = torch._grouped_mm(x, w1_t, offs=offs)  # pylint: disable=protected-access
    h3 = torch._grouped_mm(x, w3_t, offs=offs)  # pylint: disable=protected-access
    h = F.silu(h1) * h3
    return torch._grouped_mm(h, w2_t, offs=offs)  # pylint: disable=protected-access


def _run_experts_grouped_mm_npu(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Fused grouped matmul path for Ascend NPU using ``torch_npu.npu_grouped_matmul``.

    Requires ``torch_npu`` to be installed.

    Args:
        w1: Shape ``[num_experts, hidden_dim, dim]``.
        w2: Shape ``[num_experts, dim, hidden_dim]``.
        w3: Shape ``[num_experts, hidden_dim, dim]``.
        x: Shape ``[total_routed_tokens, dim]``.
        num_tokens_per_expert: 1-D integer tensor of length ``num_experts``.

    Returns:
        Expert output of shape ``[total_routed_tokens, dim]``.
    """
    import torch_npu  # pylint: disable=C0415

    # npu_grouped_matmul computes y = x @ weight (no implicit transpose).
    # Our weight storage is [num_experts, out_dim, in_dim], matching F.linear's
    # convention (weight.T for y = x @ weight.T). Transpose each expert shard
    # so the shapes satisfy: [tokens, in_dim] @ [in_dim, out_dim] = [tokens, out_dim].
    num_experts = w1.shape[0]
    counts = num_tokens_per_expert.tolist()
    x_list = list(torch.split(x, counts, dim=0))
    # w1, w3: [E, hidden_dim, dim]  → transposed per-expert: [dim, hidden_dim]
    # w2:     [E, dim, hidden_dim]  → transposed per-expert: [hidden_dim, dim]
    w1_list = [w1[e].T.contiguous() for e in range(num_experts)]
    w2_list = [w2[e].T.contiguous() for e in range(num_experts)]
    w3_list = [w3[e].T.contiguous() for e in range(num_experts)]

    # npu_grouped_matmul: multi-multi-multi mode (x[i] @ weight[i]).
    # group_type=-1 selects independent per-expert matmul (no shared axis).
    h1_list = torch_npu.npu_grouped_matmul(x_list, w1_list, group_type=-1)
    h3_list = torch_npu.npu_grouped_matmul(x_list, w3_list, group_type=-1)
    h_list = [F.silu(h1) * h3 for h1, h3 in zip(h1_list, h3_list)]
    out_list = torch_npu.npu_grouped_matmul(h_list, w2_list, group_type=-1)
    return torch.cat(out_list, dim=0)


# ---------------------------------------------------------------------------
# FeedForward — shared expert / standard SwiGLU FFN
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU feed-forward network, used as a shared (always-active) expert.

    Implements: ``output = w2(silu(w1(x)) * w3(x))``

    Args:
        dim: Input embedding dimension.
        hidden_dim: Intermediate hidden dimension.
        bias: Whether to add a learnable bias.  Defaults to ``False``.

    Example::
        >>> ff = FeedForward(dim=256, hidden_dim=512)
        >>> out = ff(torch.randn(4, 16, 256))
        >>> out.shape
        torch.Size([4, 16, 256])
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU feed-forward output.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor with the same leading shape and last dimension ``dim``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# GroupedExperts
# ---------------------------------------------------------------------------

class GroupedExperts(nn.Module):
    """Batch expert computation with optional grouped matrix-multiply.

    All expert weights are stored in a single 3-D parameter so that EP / TP
    sharding strategies can distribute the expert dimension via DTensor.

    Args:
        dim: Token embedding dimension.
        hidden_dim: Expert hidden dimension (SwiGLU intermediate size).
        num_experts: Total number of experts.
        use_grouped_mm: If ``True``, uses a hardware-accelerated grouped
            matmul kernel (``torch._grouped_mm`` on GPU,
            ``torch_npu.npu_grouped_matmul`` on NPU).  Falls back to the
            for-loop path when neither is available.  Defaults to ``False``.

    Note:
        When weights are DTensors (e.g. after TP sharding via
        :class:`ExpertTensorParallel`), ``forward`` calls ``.to_local()``
        before computation.

    Example::
        >>> experts = GroupedExperts(dim=8, hidden_dim=16, num_experts=4)
        >>> x = torch.randn(10, 8)
        >>> counts = torch.tensor([3, 2, 4, 1])
        >>> out = experts(x, counts)
        >>> out.shape
        torch.Size([10, 8])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = False,
    ) -> None:
        super().__init__()
        # Weight layout: [num_experts, out_dim, in_dim] so that the standard
        # linear operation is x @ w[e].T.
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.num_experts = num_experts
        self.use_grouped_mm = use_grouped_mm
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming-uniform initialisation for all expert weight tensors."""
        for weight in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(weight.view(weight.shape[0], -1), a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run all experts on their assigned tokens.

        Args:
            x: Routed tokens in expert-major order,
                shape ``[total_routed_tokens, dim]``.
            num_tokens_per_expert: 1-D integer tensor of length
                ``num_local_experts`` with the token count per expert.

        Returns:
            Expert output of shape ``[total_routed_tokens, dim]``.
        """
        # Extract local shard when parameters are DTensors (TP path).
        w1 = self.w1.to_local() if isinstance(self.w1, DTensor) else self.w1
        w2 = self.w2.to_local() if isinstance(self.w2, DTensor) else self.w2
        w3 = self.w3.to_local() if isinstance(self.w3, DTensor) else self.w3

        if not self.use_grouped_mm:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

        if hasattr(torch, 'npu') and torch.npu.is_available():
            return _run_experts_grouped_mm_npu(w1, w2, w3, x, num_tokens_per_expert)
        if torch.cuda.is_available():
            return _run_experts_grouped_mm_gpu(w1, w2, w3, x, num_tokens_per_expert)

        return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)


# ---------------------------------------------------------------------------
# TokenChoiceTopKRouter
# ---------------------------------------------------------------------------

class TokenChoiceTopKRouter(nn.Module):
    """Top-K router: each token independently selects its top-K experts.

    Args:
        dim: Token embedding dimension (input to the gate).
        num_experts: Total number of experts.
        top_k: Experts selected per token.  Defaults to ``1``.
        score_func: Activation on gate logits before topk.
            One of ``"sigmoid"`` (default) or ``"softmax"``.
        num_expert_groups: For node-limited routing, number of expert
            groups.  ``None`` disables node-limited routing.
        num_limited_groups: Groups to keep in node-limited routing.  Required
            when ``num_expert_groups`` is not ``None``.
        route_scale: Scalar multiplier applied to routing scores.
            Defaults to ``1.0``.

    Example::
        >>> router = TokenChoiceTopKRouter(dim=64, num_experts=8, top_k=2)
        >>> scores, indices, counts = router(torch.randn(32, 64))
        >>> scores.shape, indices.shape, counts.shape
        (torch.Size([32, 2]), torch.Size([32, 2]), torch.Size([8]))
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 1,
        score_func: str = "sigmoid",
        num_expert_groups: Optional[int] = None,
        num_limited_groups: Optional[int] = None,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if score_func not in ("sigmoid", "softmax"):
            raise ValueError(
                f"score_func must be 'sigmoid' or 'softmax', got '{score_func}'."
            )
        if num_expert_groups is not None and num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is not None."
            )
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.route_scale = route_scale

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: Optional[torch.Tensor] = None,
    ):
        """Compute routing scores and top-K expert assignments.

        Args:
            x: Token tensor of shape ``[num_tokens, dim]``.
            expert_bias: Optional 1-D tensor of shape ``[num_experts]`` added
                to gate logits for topk selection only (auxiliary-loss-free
                load balancing).  Does not affect the returned ``top_scores``.

        Returns:
            Tuple of:

            - ``top_scores``: shape ``[num_tokens, top_k]`` — routing weights.
            - ``selected_experts``: shape ``[num_tokens, top_k]`` — expert IDs.
            - ``num_tokens_per_expert``: shape ``[num_experts]`` — load counts.
        """
        # Gate in float32 for numerical stability.
        scores = self.gate(x).float()

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        else:
            scores = F.softmax(scores, dim=-1)

        if self.route_scale != 1.0:
            scores = scores * self.route_scale

        # Node-limited routing — mask out low-scoring expert groups.
        scores_for_topk = scores
        if self.num_expert_groups is not None:
            scores_for_topk = self._get_node_limited_routing_scores(scores)

        # Add expert bias only for selection; returned scores remain unbiased.
        scores_with_bias = scores_for_topk
        if expert_bias is not None:
            scores_with_bias = scores_for_topk + expert_bias.float()

        top_scores, selected_experts = scores_with_bias.topk(self.top_k, dim=-1)
        # Gather unbiased scores as the actual routing weights.
        top_scores = scores.gather(1, selected_experts)

        num_tokens_per_expert = torch.bincount(
            selected_experts.flatten(), minlength=self.num_experts
        )
        return top_scores, selected_experts, num_tokens_per_expert

    def _get_node_limited_routing_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Mask out low-scoring expert groups (node-limited routing).

        Args:
            scores: Routing scores of shape ``[num_tokens, num_experts]``.

        Returns:
            Scores with non-selected groups masked to ``-inf``.
        """
        num_tokens, num_experts = scores.shape
        experts_per_group = num_experts // self.num_expert_groups
        group_scores = scores.view(
            num_tokens, self.num_expert_groups, experts_per_group
        ).max(dim=-1).values  # [num_tokens, num_groups]

        _, selected_groups = group_scores.topk(self.num_limited_groups, dim=-1)

        mask = scores.new_zeros(num_tokens, self.num_expert_groups)
        mask.scatter_(1, selected_groups, 1.0)
        mask = (
            mask.unsqueeze(-1)
            .expand(num_tokens, self.num_expert_groups, experts_per_group)
            .reshape(num_tokens, num_experts)
        )
        return scores.masked_fill(mask == 0, float("-inf"))


# ---------------------------------------------------------------------------
# Load-balance auxiliary loss
# ---------------------------------------------------------------------------

def _compute_load_balance_loss(
    top_scores: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute load-balance auxiliary loss.

    Standard formulation: ``loss = num_experts * Σ fraction_i * mean_score_i``

    Args:
        top_scores: Routing weights, shape ``[num_tokens, top_k]``.
        selected_experts: Expert IDs, shape ``[num_tokens, top_k]``.
        num_experts: Total number of experts.

    Returns:
        Scalar loss tensor.
    """
    num_tokens, top_k = top_scores.shape
    flat_experts = selected_experts.flatten()  # [num_tokens * top_k]

    # Fraction of tokens sent to each expert (soft, uses routing probabilities).
    one_hot = torch.zeros(
        num_tokens * top_k, num_experts,
        dtype=top_scores.dtype, device=top_scores.device,
    ).scatter_(1, flat_experts.unsqueeze(1), 1.0)
    # [num_tokens, top_k, num_experts] → mean per expert
    expert_fraction = one_hot.view(num_tokens, top_k, num_experts).float().mean(dim=(0, 1))

    # Mean routing probability per expert.
    prob = F.softmax(top_scores.float(), dim=-1)  # [num_tokens, top_k]
    mean_score = prob.mean(dim=0)  # [top_k]

    # Scalar: num_experts * dot(mean_score, expert_fraction.T)
    loss = num_experts * (mean_score.unsqueeze(-1) * expert_fraction.unsqueeze(0)).sum()
    return loss


# ---------------------------------------------------------------------------
# MoE orchestrator
# ---------------------------------------------------------------------------

class MoE(nn.Module):
    """Mixture-of-Experts layer.

    Orchestrates routing, token permutation, expert computation, and output
    scatter-add.  Supports shared experts, auxiliary-loss-free load balancing
    via expert bias, node-limited routing, and auxiliary load-balance loss.

    Args:
        dim: Token embedding dimension.
        hidden_dim: Expert hidden dimension.
        num_experts: Total number of experts.
        top_k: Experts selected per token.  Defaults to ``1``.
        score_before_experts: If ``True``, multiply routed tokens by routing
            weights *before* expert computation; otherwise multiply expert
            outputs *after*.  Defaults to ``True``.
        load_balance_coeff: When not ``None``, attaches an auxiliary load-
            balance loss as ``output._load_balance_loss``.
        shared_expert: Optional :class:`FeedForward` running on every token
            in parallel; output added to routed-expert output.
        router_kwargs: Extra keyword arguments forwarded to
            :class:`TokenChoiceTopKRouter`.
        use_grouped_mm: If ``True``, uses a hardware-accelerated grouped
            matmul kernel (e.g. ``npu_grouped_matmul``) inside
            :class:`GroupedExperts`.  Defaults to ``False``.

    Note:
        *Auxiliary-loss-free load balancing*: call :func:`update_expert_bias`
        once per optimiser step to adjust ``expert_bias`` from the accumulated
        ``tokens_per_expert`` histogram.

    Example::
        >>> moe = MoE(dim=64, hidden_dim=128, num_experts=8, top_k=2)
        >>> out = moe(torch.randn(2, 16, 64))
        >>> out.shape
        torch.Size([2, 16, 64])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        score_before_experts: bool = True,
        load_balance_coeff: Optional[float] = None,
        shared_expert: Optional[FeedForward] = None,
        router_kwargs: Optional[dict] = None,
        use_grouped_mm: bool = False,
    ) -> None:
        super().__init__()
        router_kw = router_kwargs or {}
        self.experts = GroupedExperts(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim, num_experts=num_experts, top_k=top_k, **router_kw,
        )
        self.shared_expert = shared_expert
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_before_experts = score_before_experts
        self.load_balance_coeff = load_balance_coeff

        # Auxiliary-loss-free load-balance buffers (no gradient).
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.register_buffer("tokens_per_expert", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MoE layer.

        Args:
            x: Input tensor of shape ``[batch, seq_len, dim]``.

        Returns:
            Output tensor of shape ``[batch, seq_len, dim]``.  When
            ``load_balance_coeff`` is set, carries a ``_load_balance_loss``
            attribute with the auxiliary loss scalar.
        """
        # Extract local tensor if input arrives as DTensor (TP path).
        if isinstance(x, DTensor):
            x = x.to_local()

        bs, seq_len, dim = x.shape
        num_tokens = bs * seq_len
        x_flat = x.view(num_tokens, dim)  # [num_tokens, dim]

        # --- Routing ---
        top_scores, selected_experts, token_counts = self.router(
            x_flat, self.expert_bias
        )

        # Accumulate token histogram without creating gradient nodes.
        with torch.no_grad():
            self.tokens_per_expert.add_(token_counts.float())

        # --- Token permutation: token-major → expert-major (inline argsort) ---
        # flat_experts[i] is the expert ID for the i-th (token, top_k) slot.
        flat_experts = selected_experts.flatten()                  # [num_tokens * top_k]
        flat_indices = flat_experts.argsort(stable=True)           # expert-major permutation
        top_scores_sorted = top_scores.flatten()[flat_indices]     # [num_tokens * top_k]
        # Each entry in flat_indices maps to a position in [0, num_tokens * top_k);
        # divide by top_k to recover the original token row index.
        token_indices = flat_indices // self.top_k                 # [num_tokens * top_k]
        num_tokens_per_expert = torch.bincount(
            flat_experts, minlength=self.num_experts
        )

        # Gather routed tokens in expert-major order.
        routed_x = x_flat[token_indices]  # [num_tokens * top_k, dim]

        if self.score_before_experts:
            routed_x = routed_x * top_scores_sorted.unsqueeze(1)

        # --- Shared expert (parallel with routed experts) ---
        shared_out = None
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)

        # --- Expert computation ---
        expert_out = self.experts(routed_x, num_tokens_per_expert)

        if not self.score_before_experts:
            expert_out = expert_out * top_scores_sorted.unsqueeze(1)

        # --- Scatter expert outputs back to token order ---
        # Use out-of-place ``scatter_add`` so autograd correctly records
        # ``ScatterAddBackward``; ``new_zeros + scatter_add_`` on some
        # backends (torch_npu) leaves the leaf un-upgraded and the result
        # without a ``grad_fn``.
        out = torch.zeros(
            num_tokens, dim, dtype=x_flat.dtype, device=x_flat.device,
        ).scatter_add(
            0,
            token_indices.unsqueeze(1).expand(-1, dim),
            expert_out,
        )

        if shared_out is not None:
            out = out + shared_out

        result = out.view(bs, seq_len, dim)

        # Auxiliary load-balance loss attached to the returned tensor.
        if self.load_balance_coeff is not None:
            lb_loss = self.load_balance_coeff * _compute_load_balance_loss(
                top_scores, selected_experts, self.num_experts
            )
            result._load_balance_loss = lb_loss  # pylint: disable=protected-access

        return result


# ---------------------------------------------------------------------------
# Expert bias update for auxiliary-loss-free load balancing
# ---------------------------------------------------------------------------

def update_expert_bias(moe: MoE, lr: float = 1e-3) -> None:
    """Update expert bias for auxiliary-loss-free load balancing.

    Should be called once per training step after the optimiser step.
    Adjusts ``moe.expert_bias`` to push token load towards the mean, then
    resets the ``tokens_per_expert`` accumulator.

    Args:
        moe: The :class:`MoE` module whose bias should be updated.
        lr: Step size for the bias update.  Defaults to ``1e-3``.

    Example::
        >>> # After optimizer.step():
        >>> update_expert_bias(moe_layer, lr=1e-3)
    """
    with torch.no_grad():
        avg = moe.tokens_per_expert.float().mean()
        moe.expert_bias.data += lr * (avg - moe.tokens_per_expert.float()).sign()
        moe.tokens_per_expert.zero_()
