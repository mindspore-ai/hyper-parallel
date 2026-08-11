# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Standalone MoE Module - Extracted from Sophon-Pytorch
# This module contains the core MoE algorithms with all hardware acceleration
# features preserved, but with all distributed/memory optimization/overlap logic removed.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from hyper_parallel.models.vl_moe.model import (
        VLTextConfig,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_parallel.core.dtensor.dtensor import DTensor

from .aux_loss import AuxLossAutoScaler, switch_load_balancing_loss_func, z_loss_func
from .mlp import LinearWithMatmul

# Conditional NPU imports
try:
    import torch_npu
    HAS_NPU = True
except (ImportError, Exception):
    HAS_NPU = False


# ============================================================================
# GMM
# ============================================================================
class GMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args

        npu_bias = [bias] if bias is not None else []

        outputs = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[weight],
            bias=npu_bias,
            group_list=group_list,
            split_item=3,
            group_type=group_type,
            group_list_type=group_list_type
        )

        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight)
            ctx.group_list = group_list
        else:
            ctx.save_for_backward(x, weight, group_list)

        ctx.original_weight = original_weight
        ctx.gemm_fusion = gemm_fusion
        ctx.group_list_type = group_list_type
        ctx.group_list_data_type = group_list_data_type
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()

        if ctx.group_list_data_type == 0:
            x, weight = ctx.saved_tensors
            group_list = ctx.group_list
        else:
            x, weight, group_list = ctx.saved_tensors

        original_weight = ctx.original_weight
        group_list_type = ctx.group_list_type

        if isinstance(group_list, (list, tuple)):
            group_list_tensor = torch.tensor(group_list, device=x.device, dtype=torch.int64)
        else:
            group_list_tensor = group_list

        dx_list = torch_npu.npu_grouped_matmul(
            x=[grad_outputs],
            weight=[weight.transpose(-1, -2)],
            bias=[],
            group_list=group_list_tensor,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type
        )
        dx = dx_list[0]

        grad_weight = None
        use_fusion = False
        if ctx.gemm_fusion:
            if hasattr(original_weight, 'main_grad'):
                if original_weight.main_grad is not None:
                    use_fusion = True

        if use_fusion:
            actual_group_list = group_list_tensor

            n_size = original_weight.main_grad.shape[-1]
            main_grad_view = original_weight.main_grad.view(-1, n_size)

            torch_npu.npu_grouped_matmul_add_(
                main_grad_view,
                x,
                grad_outputs,
                actual_group_list,
                transpose_x=True,
                transpose_weight=False,
                group_type=2
            )

            if getattr(weight, 'zero_out_wgrad', False):
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = torch.empty_like(weight)

            original_weight.grad_added_to_main_grad = True
        else:
            if x.nelement() > 0:
                dw_list = torch_npu.npu_grouped_matmul(
                    x=[x.transpose(-1, -2)],
                    weight=[grad_outputs],
                    bias=[],
                    group_list=group_list_tensor,
                    split_item=2,
                    group_type=2,
                    group_list_type=group_list_type
                )
                grad_weight = dw_list[0]
            else:
                grad_weight = torch.zeros_like(weight)

        return None, dx, grad_weight, None, None

def npu_gmm(
        x,
        weight,
        bias=None,
        group_list=None,
        group_type=0,
        group_list_type=0,
        gemm_fusion=False,
        original_weight=None
    ):
    group_list_data_type = 1 if isinstance(group_list, torch.Tensor) else 0
    group_args = (group_list, group_type, gemm_fusion, group_list_type, group_list_data_type)

    if original_weight is None:
        original_weight = weight

    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


# ============================================================================
# Utility functions
# ============================================================================

def _get_act_fn(hidden_act):
    """Map activation function name to callable."""
    _ACT_NAME_TO_FN = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
        "gelu_new": F.gelu,
        "gelu_pytorch_tanh": F.gelu,
        "tanh": torch.tanh,
    }
    return _ACT_NAME_TO_FN.get(hidden_act, F.silu)




class _LinearWithoutInitialization(nn.Linear):
    """Linear layer whose parameters are initialized explicitly by its owner."""

    def reset_parameters(self) -> None:
        pass


# ============================================================================
# MoE utility functions
# ============================================================================


def _native_permute_without_pad(tokens, indices, num_out_tokens: int = None):
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def _native_unpermute_without_pad(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    topk: int = 1,
):
    """Scatter expert outputs back to original token positions by sorted_indices and accumulate.

    Aligned with unpermute_with_ep (moe_utils.py):
      1. First multiply by permuted_probs, then scatter_add_ to accumulate (order)
      2. Scatter directly to [num_tokens, hidden] instead of reshaping to [num_tokens, topk, hidden] then sum
    Under bf16, multiplication and accumulation order is not commutative, must strictly match.
    """
    if sorted_indices.numel() != permuted_tokens.size(0):
        raise ValueError(f"Got {sorted_indices.numel()} != {permuted_tokens.size(0)}.")

    if probs is not None:
        num_unpermuted_tokens = probs.size(0)
    else:
        num_unpermuted_tokens = permuted_tokens.size(0)

    input_dtype = permuted_tokens.dtype

    if probs is not None:
        flatten_probs = probs.view(-1)
        permuted_probs = flatten_probs.index_select(0, sorted_indices)
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    token_indices = sorted_indices // topk if topk > 1 else sorted_indices
    unpermuted_tokens = unpermuted_tokens.scatter_add_(
        0,
        token_indices.unsqueeze(1).expand(-1, permuted_tokens.shape[1]),
        permuted_tokens,
    )

    return unpermuted_tokens.to(input_dtype)


def compute_routing_scores(logits: torch.Tensor, score_function: str) -> torch.Tensor:
    if score_function == "softmax":
        return torch.softmax(logits, dim=-1, dtype=torch.float32)
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        return scores.to(logits.dtype)
    raise ValueError(f"Invalid score_function: {score_function}")


def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    norm_topk_prob: bool = False,
):
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}.")
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            if expert_bias is not None:
                scores += expert_bias
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            if expert_bias is not None:
                logits = logits + expert_bias
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if topk > 1 and norm_topk_prob:
        denominator = probs.sum(dim=-1, keepdim=True) + 1e-20
        probs = probs / denominator

    if scaling_factor:
        probs = probs * scaling_factor

    probs = probs.type_as(logits)
    return probs, top_indices


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    group_scores = scores.view(num_tokens, num_groups, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


# ============================================================================
# TokenDispatcher
# ============================================================================

class TokenDispatcher:
    """Local-only token dispatcher for standalone MoE (EP=1, TP=1).

    In standalone mode, token permutation is simply:
    1. Sort tokens by expert assignment (permute)
    2. After expert computation, restore original order and merge with probs (unpermute)
    """

    def __init__(self, num_experts: int, topk: int, use_fused_permute: bool = False):
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self.topk = topk
        self.use_fused_permute = use_fused_permute

        # State saved during permutation for use in unpermutation
        self._hidden_shape = None
        self._probs = None
        self._sorted_indices = None

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
        permute_recompute_info=None,
    ):
        self._hidden_shape = hidden_states.shape
        self._probs = probs

        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_fused_permute:
            raise NotImplementedError("use_fused_permute is not supported yet.")
        else:
            permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(hidden_states_flat, indices)
        self._sorted_indices = sorted_indices

        flatten_probs = probs.view(-1)
        permuted_probs = flatten_probs.index_select(0, sorted_indices)

        tokens_per_expert = torch.histc(
            indices.view(-1).float(), bins=self.num_experts, min=0, max=self.num_experts
        ).long()

        return permuted_tokens, tokens_per_expert, permuted_probs, None

    def token_unpermutation(self, hidden_states: torch.Tensor):
        if self.use_fused_permute:
            raise NotImplementedError("use_fused_permute is not supported yet.")
        else:
            output = torch_npu.npu_moe_token_unpermute(
                hidden_states, self._sorted_indices, self._probs
            )
        output = output.view(self._hidden_shape)
        return output


# ============================================================================
# SharedExpert
# ============================================================================

class SharedExpert(nn.Module):
    def __init__(self, config: VLTextConfig):
        super().__init__()
        self.config = config
        _moe_shared_expert_hidden_size = config.moe_intermediate_size * config.n_shared_experts
        fc1_factor = 2 if config.gated_linear_unit else 1
        _activation_func = _get_act_fn(config.hidden_act)

        self.linear_fc1 = LinearWithMatmul(
            config.hidden_size, _moe_shared_expert_hidden_size * fc1_factor,
            bias=config.attention_bias,
        )
        self.linear_fc2 = LinearWithMatmul(
            _moe_shared_expert_hidden_size, config.hidden_size,
            bias=config.attention_bias,
        )

        if config.gated_linear_unit:
            if _activation_func is F.silu and config.use_fused_swiglu and HAS_NPU:
                self.activation_func = partial(torch_npu.npu_swiglu, dim=-1)
            else:
                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return _activation_func(x[0]) * x[1]
                self.activation_func = glu
        else:
            self.activation_func = _activation_func

        if config.perform_initialization:
            config._standalone_init_weights(self)
            self._is_hf_initialized = True

    def forward(self, hidden_states):
        intermediate, bias = self.linear_fc1(hidden_states)
        if bias is not None:
            intermediate = intermediate + bias
        intermediate = self.activation_func(intermediate)
        output, output_bias = self.linear_fc2(intermediate)
        return output, output_bias


# ============================================================================
# Experts base class
# ============================================================================

class Experts(nn.Module):
    """Pure weight container for routed experts.

    Holds gate_up_proj, down_proj (and optional bias1, bias2) as nn.Parameters.
    Computation logic is handled by MoELayer.
    """

    def __init__(self, config: VLTextConfig):
        super().__init__()
        self.config = config
        self.num_local_experts = config.n_routed_experts
        self.add_bias = config.attention_bias
        self.router_gating_in_fp32 = config.router_gating_in_fp32
        _activation_func = _get_act_fn(config.hidden_act)
        _params_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

        # Activation function for expert computation
        if config.gated_linear_unit:
            if _activation_func is F.silu and config.use_fused_swiglu and HAS_NPU:
                self.activation_func = partial(torch_npu.npu_swiglu, dim=-1)
            else:
                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return _activation_func(x[0]) * x[1]
                self.activation_func = glu
        else:
            self.activation_func = _activation_func

        # GroupGemm state (NPU-only)
        self._group_list = None
        self._tokens_per_expert_gmm = None

        # TP=1 in standalone mode
        _moe_intermediate_size = config.moe_intermediate_size
        if _moe_intermediate_size is None:
            _moe_intermediate_size = config.intermediate_size
        fc1_output_size = _moe_intermediate_size
        if config.gated_linear_unit:
            fc1_output_size *= 2
        fc1_output_size_per_partition = fc1_output_size

        fc2_input_size = _moe_intermediate_size
        fc2_input_size_per_partition = fc2_input_size

        self.use_2d_experts = config.experts_2d
        if self.use_2d_experts:
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    self.num_local_experts * config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=_params_dtype,
                )
            )
            self.down_proj = nn.Parameter(
                torch.empty(
                    self.num_local_experts * fc2_input_size_per_partition,
                    config.hidden_size,
                    dtype=_params_dtype,
                )
            )
        else:
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=_params_dtype,
                )
            )
            self.down_proj = nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    fc2_input_size_per_partition,
                    config.hidden_size,
                    dtype=_params_dtype,
                )
            )

        if self.add_bias:
            self.bias1 = nn.Parameter(
                torch.empty(
                    self.num_local_experts * fc1_output_size_per_partition,
                    dtype=_params_dtype,
                )
            )
            self.bias2 = nn.Parameter(
                torch.empty(
                    self.num_local_experts * config.hidden_size,
                    dtype=_params_dtype,
                )
            )
        if config.perform_initialization:
            config._standalone_init_weights(self)
            self._is_hf_initialized = True

    def _sequential_expert_forward(
        self, gate_up_proj, down_proj, permuted, tokens_per_expert, permuted_probs
    ):
        """Sequential expert computation: loop over experts one by one."""
        expert_inputs = torch.split(permuted, tokens_per_expert.tolist(), dim=0)
        # up
        up_results = []
        for i in range(self.num_local_experts):
            out = torch.matmul(expert_inputs[i], gate_up_proj[i])
            if self.add_bias:
                out = out + self.bias1.reshape(self.num_local_experts, 1, -1)[i]
            up_results.append(out)
        # down
        expert_outputs = []
        accum_token_num = 0
        for i in range(self.num_local_experts):
            expert_token_num = up_results[i].shape[0]
            if permuted_probs is not None:
                fc2_input = self.activation_func(up_results[i]) * permuted_probs[
                    accum_token_num: accum_token_num + expert_token_num
                ].unsqueeze(1)
            else:
                fc2_input = self.activation_func(up_results[i])
            accum_token_num += expert_token_num
            out = torch.matmul(fc2_input, down_proj[i])
            if self.add_bias:
                out = out + self.bias2.reshape(self.num_local_experts, 1, -1)[i]
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=0)

    def _grouped_gemm_expert_forward(
        self, gate_up_proj, down_proj, permuted, tokens_per_expert, permuted_probs
    ):
        """GroupGemm expert computation (NPU-only)."""
        self._tokens_per_expert_gmm = tokens_per_expert.to(device=permuted.device)
        self._group_list = self._tokens_per_expert_gmm.cumsum(dim=0)

        if self.use_2d_experts:
            gate_up_proj = gate_up_proj.view(self.num_local_experts, self.config.hidden_size, -1)

        # up
        if permuted.nelement() != 0:
            fc1_output = npu_gmm(
                permuted, gate_up_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
            if self.add_bias:
                b1 = self.bias1.view(self.num_local_experts, 1, -1)
                fc1_output = fc1_output + torch.repeat_interleave(b1, self._tokens_per_expert_gmm, dim=0)
        else:
            gate_up_proj_2d = gate_up_proj.view(self.config.hidden_size, -1)
            fc1_output = torch.matmul(permuted, gate_up_proj_2d)

        # down
        if not self.router_gating_in_fp32 and permuted_probs is not None:
            fc1_output = (
                self.activation_func(fc1_output)
                * permuted_probs.reshape(*fc1_output.shape[:-1], 1)
            )
        else:
            fc1_output = self.activation_func(fc1_output)

        if self.use_2d_experts:
            down_proj = down_proj.view(self.num_local_experts, -1, self.config.hidden_size)

        if fc1_output.nelement() != 0:
            fc2_output = npu_gmm(
                fc1_output, down_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
            if self.add_bias:
                b2 = self.bias2.view(self.num_local_experts, 1, -1)
                fc2_output = fc2_output + torch.repeat_interleave(
                    b2, self._tokens_per_expert_gmm, dim=0,
                )
        else:
            down_proj_2d = down_proj.view(-1, self.config.hidden_size)
            fc2_output = torch.matmul(fc1_output, down_proj_2d)

        return fc2_output

    def forward(self, x, num_tokens_per_expert, scores=None):
        gate_up_proj = (
            self.gate_up_proj.to_local()
            if isinstance(self.gate_up_proj, DTensor)
            else self.gate_up_proj
        )
        down_proj = (
            self.down_proj.to_local()
            if isinstance(self.down_proj, DTensor)
            else self.down_proj
        )
        self.num_local_experts = num_tokens_per_expert.shape[0]

        if self.config.moe_grouped_gemm:
            return self._grouped_gemm_expert_forward(
                gate_up_proj, down_proj, x, num_tokens_per_expert, scores
            )
        else:
            return self._sequential_expert_forward(
                gate_up_proj, down_proj, x, num_tokens_per_expert, scores
            )


# ============================================================================
# MoELayer
# ============================================================================

class MoELayer(nn.Module):
    def __init__(self, config: VLTextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.routing_type = config.moe_router_load_balancing_type
        self.score_function = "sigmoid" if config.moe_use_sigmoid_gating else "softmax"
        self.num_groups = config.n_group if config.n_group is not None else 1
        self.group_top_k = config.topk_group
        self.input_jitter = None

        # Match Qwen3-VL MoE's layout: the sparse block owns its bias-free gate
        # and performs routing directly.
        # Skip nn.Linear's default initialization because the legacy router
        # allocated an empty FP32 Parameter and initialized it exactly once.
        self.gate = _LinearWithoutInitialization(
            self.hidden_size, self.num_experts, bias=False
        )
        self.gate.weight = nn.Parameter(
            torch.empty((self.num_experts, self.hidden_size), dtype=torch.float32)
        )
        params_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        if config.perform_initialization:
            config._standalone_init_weights(self)
            self._is_hf_initialized = True
        self.gate.weight.data = self.gate.weight.data.to(dtype=params_dtype)
        self.gate._is_hf_initialized = True

        if config.router_enable_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
            self.register_buffer("local_tokens_per_expert", torch.zeros(self.num_experts, dtype=torch.int32), persistent=False)
        else:
            self.expert_bias = None
            self.local_tokens_per_expert = None

        if config.moe_router_scale:
            self.router_scale = nn.Parameter(torch.ones((1, config.n_routed_experts)))

        self.token_dispatcher = TokenDispatcher(
            num_experts=config.n_routed_experts,
            topk=config.num_experts_per_tok,
            use_fused_permute=config.use_fused_permute,
        )

        self.experts = Experts(config)

        _moe_shared_expert_hidden_size = (
            config.moe_intermediate_size * config.n_shared_experts
            if config.moe_intermediate_size else None
        )
        self._has_shared_expert = _moe_shared_expert_hidden_size is not None
        if self._has_shared_expert:
            self.shared_expert = SharedExpert(config)

    def get_sandwich_post_norm_scale(self):
        return self.config.attn_post_norm_scale

    def _apply_input_jitter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        eps = self.config.moe_input_jitter_eps
        if eps is None:
            return hidden_states

        if self.input_jitter is None:
            self.input_jitter = torch.distributions.uniform.Uniform(
                torch.tensor(1.0 - eps, device=hidden_states.device),
                torch.tensor(1.0 + eps, device=hidden_states.device),
            ).rsample
        return hidden_states * self.input_jitter(hidden_states.shape)

    def _compute_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.router_gating_in_fp32:
            current_device = hidden_states.device
            # NPU computation cannot align with megatron, align must use .cpu()
            logits = torch.nn.functional.linear(hidden_states.float(), self.gate.weight.float())
            return logits.to(current_device)
        return self.gate(hidden_states)

    def _route(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.moe_z_loss_coeff is not None and self.training:
            z_loss = z_loss_func(router_logits, self.config.moe_z_loss_coeff)
            router_logits = AuxLossAutoScaler.apply(router_logits, z_loss)

        if self.routing_type in ("aux_loss", "none"):
            routing_weights, router_indices = topk_routing_with_score_function(
                router_logits,
                self.top_k,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                score_function=self.score_function,
                norm_topk_prob=True,
            )
            routing_weights = routing_weights * (
                torch.zeros(1, dtype=routing_weights.dtype, device=routing_weights.device)
                + self.config.routed_scaling_factor
            )
        elif self.routing_type == "noaux_tc":
            routing_weights, router_indices = topk_routing_with_score_function(
                router_logits,
                self.top_k,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.num_groups,
                group_topk=self.group_top_k,
                scaling_factor=self.config.routed_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                norm_topk_prob=self.config.norm_topk_prob,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        if self.expert_bias is not None and torch.is_grad_enabled():
            tokens_per_expert = torch.histc(
                router_indices.float(),
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            ).long()
        else:
            tokens_per_expert = torch.histc(
                router_indices.float(), bins=self.num_experts, min=0, max=self.num_experts
            ).long()

        if self.training:
            routing_probs = compute_routing_scores(router_logits, self.score_function)
            if self.routing_type == "aux_loss":
                aux_loss = switch_load_balancing_loss_func(
                    routing_probs, tokens_per_expert, self.top_k, self.config.moe_aux_loss_coeff,
                    aux_level=self.config.moe_aux_level, micro_batch_size=self.config.micro_batch_size,
                )
                routing_weights = AuxLossAutoScaler.apply(routing_weights, aux_loss)
            elif self.routing_type == "noaux_tc":
                if self.config.moe_aux_level == "seq":
                    _, indices_origin = torch.topk(routing_probs, self.top_k, dim=1)
                    indices_origin = indices_origin.view(self.config.micro_batch_size, -1)
                    seq_tokens_per_expert = torch.stack([
                        torch.histc(x.float(), bins=self.num_experts, min=0, max=self.num_experts).long()
                        for x in indices_origin
                    ])
                    aux_loss = switch_load_balancing_loss_func(
                        routing_probs, seq_tokens_per_expert, self.top_k, self.config.moe_aux_loss_coeff,
                        aux_level="seq", micro_batch_size=self.config.micro_batch_size,
                    )
                    routing_weights = AuxLossAutoScaler.apply(routing_weights, aux_loss)
                elif self.config.moe_aux_level == "ep-group":
                    aux_loss = switch_load_balancing_loss_func(
                        routing_probs, tokens_per_expert, self.top_k, self.config.moe_aux_loss_coeff,
                        aux_level="ep-group", micro_batch_size=self.config.micro_batch_size,
                    )
                    routing_weights = AuxLossAutoScaler.apply(routing_weights, aux_loss)

        if self.expert_bias is not None and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert.add_(tokens_per_expert)

        return routing_weights, router_indices

    def forward(self, hidden_states: torch.Tensor):
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)
        router_input = self._apply_input_jitter(hidden_states_flat)
        router_logits = self._compute_router_logits(router_input)
        probs, indices = self._route(router_logits)
        if self.config.moe_router_scale:
            broadcasted_router_scale = self.router_scale.expand(probs.shape[0], -1)
            select_router_scale = torch.gather(broadcasted_router_scale, 1, indices)
            probs = probs * select_router_scale

        permuted, tokens_per_expert, permuted_probs, _ = self.token_dispatcher.token_permutation(
            hidden_states, probs, indices
        )
        expert_outputs = self.experts(permuted, tokens_per_expert, permuted_probs)
        routed_output = self.token_dispatcher.token_unpermutation(expert_outputs)

        if self._has_shared_expert:
            shared_output, shared_bias = self.shared_expert(hidden_states)
            if shared_bias is not None:
                shared_output = shared_output + shared_bias
            output = routed_output + shared_output
        else:
            output = routed_output

        return output, None
