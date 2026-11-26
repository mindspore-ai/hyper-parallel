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
"""Compute DeepSeek V4.1 learned top-k routing before training statistics."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .._dependency import require_package, require_symbol
from ..registry import ImplementationUnavailable, OpRegistry
from ..spec import OpSpec


class MoeRouterFn(Protocol):
    """Return logits, normalized routing weights and selected expert indices."""

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, vision_bias: torch.Tensor | None,
        image_mask: torch.Tensor | None, top_k: int, scoring_func: str, scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project tokens and perform expert selection with modality corrections."""


MOE_ROUTER = OpSpec[MoeRouterFn](config_key="moe_router", variant="deepseek_v41")


class MoeTopKFn(Protocol):
    """Route projected logits, returning expert indices and normalized weights."""

    def __call__(self, logits: torch.Tensor, num_topk: int, use_shared_as_routed: bool = False,
                 num_shared_experts: int = 0, routed_scaling_factor: float = 1.0, ep_rank: int = 0,
                 **options: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve the original gate's optional masks, selection bias and inference buffers."""


class NormalizeRoutingFn(Protocol):
    """Return per-token denominators and normalized FP32 weights."""

    def __call__(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Support first-order gradients from both outputs."""


class MoeTopKBackwardFn(Protocol):
    """Explicit logit VJP, including optional normalized score-sum gradients."""

    def __call__(self, scores: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                 gradient: torch.Tensor, scale: float, **options: Any) -> torch.Tensor:
        """Use logical routed indices, without shared experts or physical remapping."""


MOE_GATE = OpSpec[MoeTopKFn](config_key="moe_router", variant="topk_gate")
MOE_GATE_BACKWARD = OpSpec[MoeTopKBackwardFn](config_key="moe_router", variant="topk_gate_backward")
ROUTING_NORMALIZE = OpSpec[NormalizeRoutingFn](config_key="moe_router", variant="normalize_weights")


def _native_normalize(weights):
    denominator = weights.sum(-1) + 1e-20
    return denominator, weights / denominator.unsqueeze(-1)


def _gate_bias(options):
    bias = options.get("bias")
    if options.get("image_token_mask") is not None:
        if options.get("image_bias") is None:
            raise ValueError("image_token_mask requires image_bias")
        bias = torch.where(options["image_token_mask"][:, None], options["image_bias"],
                           torch.zeros_like(options["image_bias"]) if bias is None else bias)
    return bias


def _native_gate(logits, num_topk, use_shared_as_routed=False, num_shared_experts=0,
                 routed_scaling_factor=1.0, ep_rank=0, **options):
    """Reference deterministic routing, fixed routes, shared experts and physical mapping."""
    if options.get("force_random") is not None:
        raise NotImplementedError("native routing does not emulate TileKernels' device RNG")
    scores = router_scores(logits, options.get("scoring_func", "sqrtsoftplus"))
    bias = _gate_bias(options)
    indices = (scores if bias is None else scores + bias).topk(
        num_topk, dim=-1, sorted=options.get("sorted_indices", False)).indices
    unmapped = options.get("unmapped_topk_idx")
    if options.get("fix_routing_mask") is not None:
        if unmapped is None:
            raise ValueError("fixed routing requires unmapped_topk_idx")
        indices = torch.where(options["fix_routing_mask"][:, None], unmapped, indices)
    weights = scores.gather(-1, indices.clamp_min(0))
    if options.get("normalize_topk", True):
        weights = _native_normalize(weights)[1]
    weights = weights * routed_scaling_factor
    if unmapped is not None:
        unmapped.copy_(indices)
    if use_shared_as_routed:
        shared = torch.arange(num_shared_experts, device=logits.device) + logits.shape[-1]
        indices = torch.cat((indices, shared.expand(logits.shape[0], -1)), dim=-1)
        weights = torch.cat((weights, weights.new_ones(weights.shape[0], num_shared_experts)), dim=-1)
    if options.get("to_physical_map") is not None:
        logical_count = options["logical_count"][indices]
        seed = ep_rank + torch.arange(logits.shape[0], device=logits.device)[:, None] * 23333
        indices = options["to_physical_map"][indices, seed.remainder(logical_count).long()].long()
    if options.get("mask") is not None:
        valid = options["mask"].reshape(-1, 1)
        indices, weights = indices.masked_fill(~valid, -1), weights.masked_fill(~valid, 0)
        if unmapped is not None:
            unmapped.masked_fill_(~valid, -1)
    if options.get("scores") is not None:
        options["scores"].copy_(scores.detach())
    if options.get("out") is not None:
        out_indices, out_weights = options["out"]
        out_indices.copy_(indices)
        out_weights.copy_(weights)
        return out_indices, out_weights
    return indices, weights


@torch.no_grad()
def _native_gate_backward(scores, indices, weights, gradient, scale, **options):
    """Recover sqrt-softplus derivatives from scores and form the dense logit VJP."""
    if options.get("scoring_func", "sqrtsoftplus") != "sqrtsoftplus":
        raise ValueError("gate backward requires sqrtsoftplus scores")
    safe = indices.clamp_min(0)
    selected = scores.gather(-1, safe).masked_fill(indices < 0, 0)
    denominator = selected.sum(-1, keepdim=True) + 1e-20
    selected_grad = (scale * gradient - (gradient * weights).sum(-1, keepdim=True)) / denominator
    dense = torch.zeros_like(scores).scatter_add_(-1, safe, selected_grad.masked_fill(indices < 0, 0))
    mask, score_grad = options.get("mask"), options.get("grad_scores_sum")
    if (mask is None) != (score_grad is None):
        raise ValueError("mask and grad_scores_sum must be provided together")
    if score_grad is not None:
        grad = score_grad.repeat_interleave(scores.shape[0] // score_grad.shape[0], dim=0)
        total = scores.sum(-1, keepdim=True)
        aux = (grad - (grad * scores).sum(-1, keepdim=True) / total) / total
        dense = dense + aux.masked_fill(~mask[:, None], 0)
    derivative = torch.where(scores > 0, -torch.expm1(-scores.square()) / (2 * scores), 0)
    result = dense * derivative
    if options.get("out") is not None:
        options["out"].copy_(result)
        return options["out"]
    return result


class _TileGate(torch.autograd.Function):
    """Connect the fused gate's supported logical-routing forward/backward pair."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: Any,
        top_k: Any,
        shared: Any,
        shared_count: Any,
        scale: Any,
        rank: Any,
        options: Any,
        kernels: Any,
    ) -> Any:
        """Execute the selected backend and retain the tensors needed for backward."""
        options = dict(options)
        scores = options.pop("scores", None)
        if scores is None:
            scores = torch.empty_like(logits)
        logical = options.get("unmapped_topk_idx")
        if logical is None and any(options.get(name) is not None for name in ("to_physical_map", "force_random")):
            logical = torch.empty(logits.shape[0], top_k, dtype=torch.int64, device=logits.device)
            options["unmapped_topk_idx"] = logical
        indices, weights = kernels.moe_topk_gate(logits, top_k, shared, shared_count, scale, rank,
                                                 scores=scores, **options)
        if logical is None:
            logical = indices[:, :top_k]
        ctx.save_for_backward(scores, logical.contiguous(), weights[:, :top_k].contiguous(), options.get("mask"))
        ctx.scale, ctx.kernel = scale, kernels.moe_topk_gate_backward
        ctx.top_k = top_k
        ctx.mark_non_differentiable(indices)
        return indices, weights

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, _indices_gradient: Any, gradient: Any) -> Any:
        """Return the input gradients required by the operator contract."""
        scores, indices, weights, mask = ctx.saved_tensors
        options = {} if mask is None else {"mask": mask, "grad_scores_sum": scores.new_zeros(1, scores.shape[-1])}
        result = ctx.kernel(scores, indices, weights, gradient[:, :ctx.top_k].contiguous(), ctx.scale, **options)
        return result, None, None, None, None, None, None, None


class _TileNormalize(torch.autograd.Function):
    """Retain both outputs' gradient semantics around the upstream raw normalization."""

    @staticmethod
    def forward(ctx: Any, weights: Any, kernel: Any) -> Any:
        """Execute the selected backend and retain the tensors needed for backward."""
        denominator, normalized = kernel(weights)
        ctx.save_for_backward(denominator, normalized)
        return denominator, normalized

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, denominator_gradient: Any, gradient: Any) -> Any:
        """Return the input gradients required by the operator contract."""
        denominator, normalized = ctx.saved_tensors
        result = (gradient - (gradient * normalized).sum(-1, keepdim=True)) / denominator[:, None]
        return result + denominator_gradient[:, None], None


def _load_gate() -> MoeTopKFn:
    kernels = require_package("tile_kernels.moe")
    for name in ("moe_topk_gate", "moe_topk_gate_backward"):
        require_symbol(kernels, name)

    def run(logits: Any, num_topk: Any, use_shared_as_routed: Any=False, num_shared_experts: Any=0,
            routed_scaling_factor: Any=1.0, ep_rank: Any=0, **options: Any) -> Any:
        """Keep the full forward interface; reject modes without an upstream training backward."""
        if not options.pop("normalize_topk", True):
            raise ValueError("TileKernels fused gate requires normalized routing weights")
        options.pop("sorted_indices", None)
        training = torch.is_grad_enabled() and logits.requires_grad
        if not training:
            return kernels.moe_topk_gate(logits, num_topk, use_shared_as_routed, num_shared_experts,
                                          routed_scaling_factor, ep_rank, **options)
        if options.get("out") is not None:
            raise ValueError("TileKernels gate training does not support preallocated output buffers")
        return _TileGate.apply(logits, num_topk, use_shared_as_routed, num_shared_experts,
                                routed_scaling_factor, ep_rank, options, kernels)

    return run


def _load_gate_backward() -> MoeTopKBackwardFn:
    kernels = require_package("tile_kernels.moe")
    require_symbol(kernels, "moe_topk_gate_backward")
    return kernels.moe_topk_gate_backward


def _load_normalize() -> NormalizeRoutingFn:
    kernels = require_package("tile_kernels.moe")
    require_symbol(kernels, "normalize_weight")
    return lambda weights: _TileNormalize.apply(weights, kernels.normalize_weight)


def router_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Convert router logits to the released nonnegative expert scores."""
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()  # pylint: disable=not-callable
    if scoring_func == "softmax":
        return logits.softmax(dim=-1)
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"Unsupported V4.1 router scoring function: {scoring_func!r}")


def _torch_router(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, vision_bias: torch.Tensor | None,
    image_mask: torch.Tensor | None, top_k: int, scoring_func: str, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute expert logits, modality corrections and normalized routing weights."""
    logits = F.linear(x.reshape(-1, x.shape[-1]).float(), weight.float())  # pylint: disable=not-callable
    scores = router_scores(logits, scoring_func)
    if image_mask is not None and vision_bias is not None:
        bias = torch.where(image_mask.reshape(-1, 1), vision_bias.unsqueeze(0), bias.unsqueeze(0))
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, weights * scale, indices


def _load_megagate() -> MoeRouterFn:
    """Load the optional fused MegaGate router for the shared routing contract."""
    require_package("hyper_multicore")
    # Only the selected MegaGate backend needs the standalone native package.
    # pylint: disable=import-outside-toplevel
    from hyper_multicore._loader import NativeComponentUnavailableError
    from hyper_multicore.modules.mega_gate.function import mega_gate
    from hyper_multicore.modules.mega_gate.module import _plan_for_device
    from hyper_multicore.torch.ops import _load_native
    # pylint: enable=import-outside-toplevel
    try:
        _load_native()
    except NativeComponentUnavailableError as error:
        raise ImplementationUnavailable(str(error)) from error

    def route(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, vision_bias: torch.Tensor | None,
        image_mask: torch.Tensor | None, top_k: int, scoring_func: str, scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply MegaGate using the shared learned routing contract."""
        del scoring_func
        return mega_gate(
            x, weight, bias, _plan_for_device(x.device), vision_bias=vision_bias,
            image_mask=None if image_mask is None else image_mask.reshape(-1).contiguous(),
            top_k=top_k, routed_scaling_factor=scale,
        )

    return route


def _load_tile_kernels() -> MoeRouterFn:
    kernels = require_package("tile_kernels.moe")
    require_symbol(kernels, "topk_gate")

    def route(
        x: Any,
        weight: Any,
        bias: Any,
        vision_bias: Any,
        image_mask: Any,
        top_k: Any,
        scoring_func: Any,
        scale: Any,
    ) -> Any:
        """Fuse selection only; retain Torch's learned scoring and gradient path."""
        logits = F.linear(x.reshape(-1, x.shape[-1]).float(), weight.float())  # pylint: disable=not-callable
        scores = router_scores(logits, scoring_func)
        if image_mask is not None and vision_bias is not None:
            bias = torch.where(image_mask.reshape(-1, 1), vision_bias.unsqueeze(0), bias.unsqueeze(0))
        indices = kernels.topk_gate((scores + bias).contiguous(), top_k)
        weights = scores.gather(1, indices)
        if top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
        return logits, weights * scale, indices

    return route


def register_moe_router(registry: OpRegistry) -> None:
    """Register DeepSeek's Torch router and MegaGate route kernel."""
    registry.register_family("moe_router", specs=(MOE_ROUTER, MOE_GATE, MOE_GATE_BACKWARD, ROUTING_NORMALIZE))
    registry.register_impl(MOE_ROUTER, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _torch_router)
    registry.register_impl(MOE_ROUTER, name="megagate", devices=("npu",), load=_load_megagate)
    registry.register_impl(MOE_ROUTER, name="tile_kernels", devices=("npu",), load=_load_tile_kernels)
    for spec, native, loader in ((MOE_GATE, _native_gate, _load_gate),
                                 (MOE_GATE_BACKWARD, _native_gate_backward, _load_gate_backward),
                                 (ROUTING_NORMALIZE, _native_normalize, _load_normalize)):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="tile_kernels", devices=("npu",), load=loader)
