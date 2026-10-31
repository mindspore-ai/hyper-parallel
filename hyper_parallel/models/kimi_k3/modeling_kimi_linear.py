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
"""Differentiable K3 text architecture with an explicit portable training path.

The public K3 architecture and checkpoint names are described by
https://huggingface.co/moonshotai/Kimi-K3 and https://arxiv.org/abs/2607.24653.
This implementation composes HyperParallel's KDA equations with independently
written training MoE, NoPE MLA, SiTU, and block attention residuals. It does not
load or execute the inference-only remote model implementation.
"""
from __future__ import annotations

# Native Torch/HF models are outside the removed platform abstraction.
# pylint: disable=forbidden-backend-import

from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from hyper_parallel.components.functional.linear_attention_packed import (
    PackedSequenceInfo, packed_causal_conv1d, resolve_packed_sequence,
)
from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput, chunked_cross_entropy
from hyper_parallel.components.modules.kimi_delta_attention import (
    KimiDeltaAttention as ReferenceDeltaAttention,
    KimiRMSNormGated,
)
from hyper_parallel.models.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
from hyper_parallel.models.kimi_k3.quantile_balancing import QuantileBalancer
from hyper_parallel.components.functional.attention_residuals import block_attention_residuals
from hyper_parallel.components.functional.activation_situ import situ_packed
from hyper_parallel.models.kimi_k3.adapter.kda import kda, short_convolution
from hyper_parallel.models.kimi_k3.adapter.attention import npu_attention
from hyper_parallel.models.kimi_k3.adapter.experts import grouped_experts


class SituAndMul(nn.Module):
    """FP32 SiTU gate/up product with the K3 clipping scales."""

    def __init__(self, beta: float = 4.0, linear_beta: float = 25.0, backend: str = "eager") -> None:
        """Set positive gate and linear clipping scales."""
        super().__init__()
        if beta <= 0 or linear_beta <= 0:
            raise ValueError("SiTU clipping scales must be positive")
        self.beta = beta
        self.linear_beta = linear_beta
        self.backend = backend

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Consume concatenated gate/up projections and retain input dtype.

        Args:
            hidden_states: Paired gate and up projections concatenated along the
                final dimension, with shape [..., 2 * intermediate_size].

        Returns:
            SiTU gate/up products with shape [..., intermediate_size].
        """
        return situ_packed(hidden_states, beta=self.beta, linear_beta=self.linear_beta, backend=self.backend)


class KimiRMSNorm(nn.Module):
    """RMSNorm using FP32 statistics and the released scale parameter name."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Create a unit scale for the final hidden dimension."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32, then apply the released activation/scale cast ordering.

        Args:
            hidden_states: Activations whose final dimension matches the scale.

        Returns:
            Values with the input shape, normalized over the final dimension and
            multiplied by the learnable scale after casting the normalized values.
        """
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return values.to(hidden_states.dtype) * self.weight


def _situ(config: KimiLinearConfig) -> SituAndMul:
    return SituAndMul(config.activation_situ_beta, config.activation_situ_linear_beta, config.situ_backend)


class KimiMLP(nn.Module):
    """Dense/shared SiTU MLP using the released gate/up/down names."""

    def __init__(self, config: KimiLinearConfig, intermediate_size: int | None = None) -> None:
        """Construct a decoder-width MLP or the combined shared experts."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _situ(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the dense or shared-expert SiTU MLP.

        Args:
            hidden_states: Decoder activations with shape [..., hidden_size].

        Returns:
            Down-projected SiTU values with the same shape as hidden_states.
        """
        projected = torch.cat((self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1)
        return self.down_proj(self.act_fn(projected))


class KimiBlockSparseMLP(nn.Module):
    """One routed latent-space expert with checkpoint-compatible w1/w2/w3."""

    def __init__(self, config: KimiLinearConfig, hidden_size: int) -> None:
        """Construct the gate, down, and up expert matrices."""
        super().__init__()
        self.w1 = nn.Linear(hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, config.moe_intermediate_size, bias=False)
        self.act_fn = _situ(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate an expert, including empty token batches.

        Args:
            hidden_states: Assigned token slots with shape [slots, expert_width],
                where expert_width is the latent or full decoder width.

        Returns:
            Expert outputs in the same slot order and shape as hidden_states.
        """
        return self.w2(self.act_fn(torch.cat((self.w1(hidden_states), self.w3(hidden_states)), dim=-1)))


class KimiMoEGate(nn.Module):
    """Sigmoid routing with selection-only correction and stepwise QB state."""

    def __init__(self, config: KimiLinearConfig) -> None:
        """Keep the released correction-bias FQN as non-optimizer state."""
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        self.quantile_balancer = (
            QuantileBalancer(self.num_experts, self.top_k, num_bins=config.quantile_num_bins,
                             ema_decay=config.quantile_ema_decay,
                             token_chunk_size=config.quantile_token_chunk_size) if config.quantile_balance else None
        )
        self.reset_parameters()

    def _apply(self, fn: Callable, recurse: bool = True) -> "KimiMoEGate":
        """Apply module-wide tensor conversions while retaining FP32 selection bias.

        Args:
            fn: Tensor conversion applied by module device or dtype operations.
            recurse: Whether to convert child modules as well as this router.

        Returns:
            This router after conversion, with correction bias remaining in FP32.
        """
        def preserve_bias_dtype(tensor: torch.Tensor) -> torch.Tensor:
            """Keep routing correction in FP32 when moving or casting the module.

            Args:
                tensor: Parameter or buffer supplied by the module conversion visitor.

            Returns:
                The converted tensor, retaining FP32 for the selection-bias buffer.
            """
            converted = fn(tensor)
            if tensor is self.e_score_correction_bias and converted.dtype != torch.float32:
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(preserve_bias_dtype, recurse=recurse)

    def reset_parameters(self) -> None:
        """Initialize every parameter and persistent routing statistic after meta."""
        nn.init.normal_(self.weight, std=self.config.initializer_range)
        nn.init.zeros_(self.e_score_correction_bias)
        if self.quantile_balancer is not None:
            self.quantile_balancer.reset_parameters()

    def forward(self, hidden_states: torch.Tensor, valid_token_mask: torch.Tensor | None = None,
                collect_router_stats: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k expert indices and differentiable unbiased mixture weights.

        Args:
            hidden_states: Full-width decoder activations [..., hidden_size]; all
                leading dimensions are flattened into routing tokens.
            valid_token_mask: Optional mask over tokens counted by QB. Invalid
                tokens still receive routing decisions; the mask excludes statistics.
            collect_router_stats: Accumulate QB observations during training when
                enabled. Recomputed forwards can disable collection.

        Returns:
            Expert indices and FP32 weights, each [tokens, top_k]. Correction bias
            affects selection only; weights use the uncorrected sigmoid scores,
            optional renormalization, and routed_scaling_factor.
        """
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = F.linear(flattened.float(), self.weight.float()).sigmoid()  # pylint: disable=not-callable
        if self.training and collect_router_stats and self.quantile_balancer is not None:
            valid = valid_token_mask.reshape(-1).bool() if valid_token_mask is not None else None
            self.quantile_balancer.accumulate(scores, self.e_score_correction_bias, valid_token_mask=valid)
        indices = (scores + self.e_score_correction_bias.float()).topk(self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(-1, indices)
        if self.top_k > 1 and self.config.moe_renormalize:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return indices, weights * self.config.routed_scaling_factor

    def finalize_router_step(self, process_groups: tuple = ()) -> Any:
        """Pool accumulated tokens and update the next step's selection bias.

        Args:
            process_groups: Orthogonal token-ownership groups over which QB pools
                observations; replicated observations must not be counted twice.

        Returns:
            Whether a nonempty global token batch updated the selection bias, or
            None when QB is disabled.
        """
        if self.quantile_balancer is not None:
            return self.quantile_balancer.finalize_step(self.e_score_correction_bias, process_groups=process_groups)
        return None


class KimiSparseMoeBlock(nn.Module):
    """Stable LatentMoE with autograd through routing weights and all experts."""

    def __init__(self, config: KimiLinearConfig) -> None:
        """Route from the full hidden state, then dispatch in latent space."""
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.use_latent_moe = config.routed_expert_hidden_size is not None
        self.moe_hidden_size = config.routed_expert_hidden_size or config.hidden_size
        self.latent_moe_use_norm = config.latent_moe_use_norm
        self.gate = KimiMoEGate(config)
        self.experts = nn.ModuleList([KimiBlockSparseMLP(config, self.moe_hidden_size)
                                      for _ in range(self.num_experts)])
        if config.num_shared_experts:
            self.shared_experts = KimiMLP(config, config.moe_intermediate_size * config.num_shared_experts)
        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(config.hidden_size, self.moe_hidden_size, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, config.hidden_size, bias=False)
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(self.moe_hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, valid_token_mask: torch.Tensor | None = None,
                collect_router_stats: bool = True) -> torch.Tensor:
        """Combine expert values before the single latent normalization/up projection.

        Args:
            hidden_states: Full-width decoder activations [batch, sequence, hidden].
            valid_token_mask: Optional [batch, sequence] mask excluding padding
                from QB statistics; it does not drop routed expert slots.
            collect_router_stats: Whether this forward contributes QB observations.

        Returns:
            Routed outputs plus any shared-expert branch, in the input shape.
        """
        indices, weights = self.gate(hidden_states, valid_token_mask, collect_router_stats)
        latent = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.use_latent_moe:
            latent = self.routed_expert_down_proj(latent)
        if self.config.moe_backend == "grouped":
            combined = grouped_experts(latent, indices, weights, self.experts, situ_backend=self.config.situ_backend,
                                       beta=self.config.activation_situ_beta,
                                       linear_beta=self.config.activation_situ_linear_beta)
        else:
            combined = torch.zeros(latent.shape, dtype=torch.float32, device=latent.device)
            for expert_idx, expert in enumerate(self.experts):
                token_idx, slot_idx = torch.where(indices == expert_idx)
                values = expert(latent[token_idx]).float() * weights[token_idx, slot_idx, None]
                combined = combined.index_add(0, token_idx, values)
            combined = combined.to(latent.dtype)
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                combined = self.routed_expert_norm(combined)
            combined = self.routed_expert_up_proj(combined)
        result = combined.reshape(hidden_states.shape)
        if self.config.num_shared_experts:
            result = result + self.shared_experts(hidden_states)
        return result


class KimiDeltaAttention(ReferenceDeltaAttention):
    """Canonical K3 KDA with optional shared kernels and unchanged parameter ownership."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        """Reuse shared parameter ownership while adapting the HF configuration."""
        options = config.linear_attn_config
        super().__init__(hidden_size=config.hidden_size, num_heads=options["num_heads"],
                         head_k_dim=options["head_dim"], head_v_dim=options["head_dim"],
                         conv_kernel_size=options["short_conv_kernel_size"],
                         gate_lower_bound=options["gate_lower_bound"],
                         use_full_rank_gate=options.get("use_full_rank_gate", True),
                         rms_norm_eps=config.rms_norm_eps, layer_idx=layer_idx)
        self.config = config
        self.reset_training_parameters()

    @torch.no_grad()
    def reset_training_parameters(self) -> None:
        """Fill constant decay values on both ordinary and FSDP-sharded parameters.

        DTensor implements fill_ directly, preserving its global layout while
        writing the same constant to every shard. Random parameter initialization
        remains on the normal DTensor RNG-dispatch path.
        """
        self.A_log.fill_(self.config.kda_a_log_init)
        self.dt_bias.fill_(self.config.kda_dt_bias_init)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                use_cache: bool = False, output_attentions: bool = False, **kwargs: Any) -> torch.Tensor:
        """Run causal KDA with explicit document resets and optional right padding.

        Padding is validated at the model boundary. No cache is returned, so the
        unused recurrent tail after right padding cannot affect another sample.

        Args:
            hidden_states: Activations [batch, sequence, hidden_size].
            attention_mask: Optional [batch, sequence] right-padding validity mask;
                invalid inputs and outputs are zeroed.
            use_cache: Must be False; recurrent inference caches are unsupported.
            output_attentions: Must be False; KDA does not return attention maps.
            **kwargs: PackedSequenceInfo or cumulative Q/K document boundaries
                consumed by resolve_packed_sequence. Any non-None cache_params or
                past_key_values argument is rejected, including empty containers.

        Returns:
            Gated, normalized and output-projected KDA values in the input shape.
        """
        if use_cache or kwargs.get("cache_params") is not None or kwargs.get("past_key_values") is not None:
            raise ValueError("K3 training KDA does not support inference caches")
        if output_attentions:
            raise ValueError("K3 training KDA does not expose attention maps")
        batch, length, _ = hidden_states.shape
        valid = attention_mask.unsqueeze(-1).to(hidden_states.dtype) if attention_mask is not None else None
        states = hidden_states if valid is None else hidden_states * valid
        head_shape = (batch, length, self.num_heads, self.head_dim)
        sequence_info = resolve_packed_sequence(kwargs, length)
        def convolution(projected: torch.Tensor, module: nn.Conv1d) -> torch.Tensor:
            """Apply the selected short convolution while preserving document boundaries.

            Args:
                projected: Q, K or V projection with shape [batch, sequence, channels].
                module: Depthwise causal convolution owning the corresponding weights.

            Returns:
                Convolved values reshaped to [batch, sequence, heads, head_dim].
            """
            if sequence_info is not None:
                if self.config.causal_conv_backend != "eager":
                    raise ValueError("K3 packed convolution currently requires causal_conv_backend='eager'")
                return packed_causal_conv1d(projected, module, sequence_info).reshape(head_shape)
            return short_convolution(
                projected, module.weight, backend=self.config.causal_conv_backend).reshape(head_shape)
        query = convolution(self.q_proj(states), self.q_conv1d)
        key = convolution(self.k_proj(states), self.k_conv1d)
        value = convolution(self.v_proj(states), self.v_conv1d)
        gate = self.f_b_proj(self.f_a_proj(states)).reshape(head_shape)
        beta = self.b_proj(states)
        output = kda(query, key, value, gate, beta, self.A_log, self.dt_bias,
                     lower_bound=self.gate_lower_bound, backend=self.config.kda_backend, sequence_info=sequence_info)
        output = self.o_norm(output, self._output_gate(states).reshape(head_shape))
        output = self.o_proj(output.reshape(batch, length, -1))
        return output if valid is None else output * valid


class KimiMLAAttention(nn.Module):
    """Causal NoPE MLA with optional fused attention and K3's sigmoid output gate."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        """Retain the released low-rank and output projection names."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = 1
        self.q_lora_rank = config.q_lora_rank
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_nope = True
        self.rotary_emb = None
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.scaling = self.q_head_dim ** -0.5
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            self.q_a_layernorm = KimiRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank,
                                   self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        """Attend to causal keys within each independent row or packed document.

        Args:
            hidden_states: Activations [batch, sequence, hidden_size].
            attention_mask: Optional [batch, sequence] right-padding validity mask.
                Padded outputs are zeroed; key masking is backend dependent.
            **kwargs: Packed document metadata accepted by resolve_packed_sequence.
                Packed execution requires batch size one. Inference cache requests
                through past_key_values or use_cache are rejected.

        Returns:
            MLA values after the optional sigmoid gate and output projection,
            with the same shape as hidden_states.
        """
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("K3 training MLA does not support inference caches")
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states) if self.config.q_lora_rank is None else self.q_b_proj(
            self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query = query.reshape(batch, length, self.num_heads, self.q_head_dim).transpose(1, 2)
        latent, shared_key = self.kv_a_proj_with_mqa(hidden_states).split(
            (self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        key_value = self.kv_b_proj(self.kv_a_layernorm(latent)).reshape(batch, length, self.num_heads, -1)
        key, value = key_value.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        shared_key = shared_key.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        key = torch.cat((key, shared_key), dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)
        sequence_info = resolve_packed_sequence(kwargs, length)
        sequence_ends = None if sequence_info is None else sequence_info.global_offsets[1:]
        if sequence_info is not None and batch != 1:
            raise ValueError("K3 packed MLA requires batch dimension one")
        allowed = None
        if self.config.mla_backend != "npu_fusion" or sequence_info is None:
            allowed = torch.ones((length, length), dtype=torch.bool, device=hidden_states.device).tril()[None, None]
            if sequence_info is not None:
                boundaries = torch.tensor(sequence_info.global_offsets[1:], device=hidden_states.device)
                documents = torch.bucketize(torch.arange(length, device=hidden_states.device), boundaries, right=True)
                allowed = allowed & (documents[:, None] == documents[None, :])[None, None]
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].bool()
        if self.config.mla_backend == "npu_fusion":
            output = npu_attention(self, query, key, value, allowed, causal=True, scaling=self.scaling,
                                   dropout=self.config.attention_dropout if self.training else 0.0,
                                   sequence_ends=sequence_ends)
            output = output.reshape(batch, length, -1)
        else:
            logits = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
            probabilities = logits.float().masked_fill(~allowed, torch.finfo(torch.float32).min).softmax(dim=-1)
            probabilities = F.dropout(probabilities, self.config.attention_dropout, self.training).to(value.dtype)
            output = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, length, -1)
        if self.use_output_gate:
            output = output * self.g_proj(hidden_states).sigmoid()
        output = self.o_proj(output)
        return output if attention_mask is None else output * attention_mask.unsqueeze(-1).to(output.dtype)


def _apply_attn_res(prefix_sum: torch.Tensor, block_residual: torch.Tensor,
                    proj: nn.Linear, norm: KimiRMSNorm, backend: str = "eager") -> torch.Tensor:
    """Mix block values using the selected backend and the layer's actual epsilon."""
    return block_attention_residuals(prefix_sum, block_residual, proj.weight, norm.weight,
                    norm_eps=norm.variance_epsilon, backend=backend)


def _residual_projection(hidden_size: int) -> nn.Linear:
    projection = nn.Linear(hidden_size, 1, bias=False)
    projection.kimi_attn_res_projection = True
    return projection


class _BlockHistoryBoundary(torch.autograd.Function):
    """Group later history gradients at block boundaries without copying storage."""

    @staticmethod
    def forward(ctx: Any, *blocks: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Expose history aliases through one boundary; no tensors are saved.

        Args:
            ctx: Autograd context retaining the per-invocation state.
        """
        ctx.set_materialize_grads(False)
        return tuple(block.view_as(block) for block in blocks)

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        """Return each block's accumulated gradient to the preceding boundary.

        Args:
            ctx: Autograd context retaining the per-invocation state.
        """
        del ctx
        return gradients


class KimiDecoderLayer(nn.Module):
    """One canonical decoder layer with explicit per-forward residual state."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        """Construct the scheduled attention and dense/latent expert branches."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_linear_attn = config.is_kda_layer(layer_idx)
        attention = KimiDeltaAttention if self.is_linear_attn else KimiMLAAttention
        self.self_attn = attention(config, layer_idx)
        if config.is_moe and layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0:
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.self_attention_res_norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.mlp_res_norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.self_attention_res_proj = _residual_projection(config.hidden_size)
            self.mlp_res_proj = _residual_projection(config.hidden_size)

    def _feed_forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
                      collect_router_stats: bool) -> torch.Tensor:
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(hidden_states, attention_mask, collect_router_stats)
        return self.mlp(hidden_states)

    # Keep the established optional arguments; history tensors must be positional for checkpoint inputs.
    # pylint: disable-next=keyword-arg-before-vararg
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                block_residual: torch.Tensor | None = None,
                collect_router_stats: bool = True,
                packed_sequence_info: PackedSequenceInfo | None = None,
                *residual_blocks: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Pass residual history and document boundaries explicitly through replay.

        Args:
            hidden_states: Current decoder activations or AttnRes prefix sum with
                shape [batch, local_sequence, hidden_size].
            attention_mask: Optional local validity mask for attention and QB.
            block_residual: Completed AttnRes blocks [batch * local_sequence,
                blocks, hidden_size], required when block attention residuals are on.
            collect_router_stats: Whether the feed-forward router accumulates QB
                observations; disable for activation replay.
            packed_sequence_info: Global document boundaries and this rank's span,
                forwarded to the scheduled KDA or MLA implementation.
            *residual_blocks: Independently stored completed blocks [batch *
                local_sequence, hidden_size] when attn_res_history_layout is blocks.
                Each is a positional Tensor input so checkpoint and swap hooks
                see every cross-layer dependency.

        Returns:
            Updated activations when AttnRes is off; otherwise the updated prefix
            sum and block history, including a new entry at a block boundary.
            The blocks layout returns each block separately to preserve shared
            storage and the swap manager's output-alias lifetime protection.
        """
        attention_options = {}
        if packed_sequence_info is not None:
            if self.is_linear_attn:
                attention_options["packed_sequence_info"] = packed_sequence_info
            else:
                attention_options.update(cu_seq_lens_q=packed_sequence_info.global_offsets,
                                         cu_seq_lens_k=packed_sequence_info.global_offsets)
        if not self.use_attn_residuals:
            hidden_states = hidden_states + self.self_attn(
                self.input_layernorm(hidden_states), attention_mask, **attention_options)
            return hidden_states + self._feed_forward(hidden_states, attention_mask, collect_router_stats)
        return self._forward_attn_res(hidden_states, attention_mask, block_residual,
                                      collect_router_stats, attention_options, residual_blocks)

    def _forward_attn_res(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
                          block_residual: torch.Tensor | None, collect_router_stats: bool,
                          attention_options: dict[str, Any], residual_blocks: tuple[torch.Tensor, ...]
                          ) -> tuple[torch.Tensor, ...]:
        """Materialize a temporary history inside the layer's recompute region."""
        prefix = hidden_states
        shape = hidden_states.shape
        separate_blocks = self.config.attn_res_history_layout == "blocks"
        if separate_blocks:
            block_residual = (torch.stack(residual_blocks, dim=1) if residual_blocks else
                              prefix.new_empty(shape[0] * shape[1], 0, shape[-1]))
        if block_residual is None:
            raise ValueError("AttnRes requires the model-owned per-forward block history")
        if block_residual.shape[1]:
            hidden_states = _apply_attn_res(prefix.reshape(-1, shape[-1]), block_residual,
                                            self.self_attention_res_proj, self.self_attention_res_norm,
                                            self.config.attn_res_backend).reshape(shape)
        boundary = self.layer_idx % self.config.attn_res_block_size == 0
        if boundary:
            if separate_blocks:
                history_dtype = torch.promote_types(block_residual.dtype, prefix.dtype)
                residual_blocks = tuple(block.to(history_dtype) for block in
                                        (*residual_blocks, prefix.reshape(-1, shape[-1])))
                residual_blocks = _BlockHistoryBoundary.apply(*residual_blocks)
                # Include this boundary's FFN in the same gradient group as later layers.
                block_residual = torch.stack(residual_blocks, dim=1)
            else:
                block_residual = torch.cat((block_residual, prefix.reshape(-1, 1, shape[-1])), dim=1)
        attention_output = self.self_attn(self.input_layernorm(hidden_states), attention_mask, **attention_options)
        prefix = attention_output if boundary else prefix + attention_output
        mlp_input = _apply_attn_res(prefix.reshape(-1, shape[-1]), block_residual,
                                    self.mlp_res_proj, self.mlp_res_norm, self.config.attn_res_backend).reshape(shape)
        prefix = prefix + self._feed_forward(mlp_input, attention_mask, collect_router_stats)
        if separate_blocks:
            return (prefix, *residual_blocks)
        return prefix, block_residual


class KimiPreTrainedModel(PreTrainedModel):
    """HF lifecycle with complete explicit initialization for training from scratch."""

    config_class = KimiLinearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["KimiDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize learned matrices and the explicit K3 gate/residual state."""
        if isinstance(module, nn.Linear):
            if self.config.attn_res_zero_init and getattr(module, "kimi_attn_res_projection", False):
                nn.init.zeros_(module.weight)
            else:
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv1d):
            module.reset_parameters()
        elif isinstance(module, (KimiRMSNorm, KimiRMSNormGated)):
            nn.init.ones_(module.weight)
        elif isinstance(module, KimiDeltaAttention):
            module.reset_training_parameters()
        elif isinstance(module, KimiMoEGate):
            module.reset_parameters()


def _context_parallel_mesh(model: nn.Module) -> Any:
    """Find the chronology used by the model's installed attention wrappers."""
    for layer in model.layers:
        mesh = getattr(layer.self_attn, "_hp_cp_mesh", None)
        if mesh is not None:
            return mesh
    return None


def _validate_cp_padding(attention_mask: torch.Tensor, cp_mesh: Any) -> None:
    """Check the global padding boundary with constant-size metadata per sample.

    A locally valid all-zero shard followed by a valid shard is globally invalid.
    All ranks inspect the same gathered metadata before entering attention.
    """
    local = torch.stack((
        ((attention_mask != 0) & (attention_mask != 1)).any(-1),
        (attention_mask[:, 1:] > attention_mask[:, :-1]).any(-1),
        attention_mask[:, 0].bool(), attention_mask[:, -1].bool(),
    ), dim=-1).to(torch.int64)
    shards = [torch.empty_like(local) for _ in range(cp_mesh.size())]
    dist.all_gather(shards, local, group=cp_mesh.get_group())
    metadata = torch.stack(shards)
    if bool(metadata[..., :2].any()) or bool((metadata[1:, :, 2] > metadata[:-1, :, 3]).any()):
        raise ValueError("K3 CP requires binary, globally right-padded sequences without interior gaps")


def _validate_training_inputs(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
                              position_ids: torch.Tensor | None, kwargs: dict[str, Any], cp_mesh: Any = None,
                              sequence_info: PackedSequenceInfo | None = None) -> None:
    """Validate global padding and positions before any attention communication."""
    if hidden_states.ndim != 3 or hidden_states.shape[1] == 0:
        raise ValueError("K3 inputs must be nonempty [batch, sequence, hidden] tensors")
    unexpected = [key for key, value in kwargs.items() if value is not None]
    if unexpected:
        raise ValueError(f"Unsupported K3 reference training inputs: {', '.join(sorted(unexpected))}")
    if attention_mask is not None:
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if cp_mesh is not None and cp_mesh.size() > 1:
            _validate_cp_padding(attention_mask, cp_mesh)
        else:
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must contain only zeros and ones")
            if bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
                raise ValueError(
                    "K3 reference training supports right padding only; left/interior padding is unsupported")
    _validate_position_ids(hidden_states, attention_mask, position_ids, sequence_info)


def _validate_position_ids(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
                           position_ids: torch.Tensor | None, sequence_info: PackedSequenceInfo | None) -> None:
    """Require chronological positions or explicit document-local resets."""
    if position_ids is not None:
        if position_ids.shape[-1] != hidden_states.shape[1]:
            raise ValueError("position_ids sequence length must match model inputs")
        if sequence_info is not None:
            positions = torch.arange(sequence_info.start, sequence_info.end, device=position_ids.device)
            boundaries = torch.tensor(sequence_info.global_offsets, device=position_ids.device)
            documents = torch.bucketize(positions, boundaries[1:], right=True)
            reset_positions = positions - boundaries[documents]
            if position_ids.shape != hidden_states.shape[:2]:
                raise ValueError("Packed position_ids must have shape [batch, sequence]")
            # The collator pads positions with zero; only real tokens must retain
            # global offsets or document resets, including an all-padding CP shard.
            valid = torch.ones_like(position_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
            global_match = ((position_ids == positions.unsqueeze(0)) | ~valid).all()
            document_match = ((position_ids == reset_positions.unsqueeze(0)) | ~valid).all()
            if not bool(global_match | document_match):
                raise ValueError("Packed position_ids must match global offsets or document-local positions")
            return
        differences = position_ids[..., 1:] - position_ids[..., :-1]
        valid = (torch.ones_like(differences, dtype=torch.bool)
                 if attention_mask is None else attention_mask[:, 1:].bool())
        if bool(((differences != 1) & valid).any()):
            raise ValueError(
                "K3 position resets require packed/document isolation, which is not implemented")


class KimiLinearModel(KimiPreTrainedModel):
    """Complete K3 text backbone, executable without any recipe replacement."""

    def __init__(self, config: KimiLinearConfig) -> None:
        """Construct the complete released layer schedule."""
        if getattr(config, "quantization_config", None):
            raise ValueError("Remove inference quantization_config for floating-point K3 pretraining construction")
        super().__init__(config)
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([KimiDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.output_attn_res_proj = _residual_projection(config.hidden_size)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text embedding module for HF multimodal composition."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the embedding table through the HF contract.

        Args:
            value: Token embedding module with the configured decoder output width.
        """
        self.embed_tokens = value

    def _resolve_training_sequence(
            self, hidden_states: torch.Tensor, packed_sequence_info: PackedSequenceInfo | None,
            cu_seq_lens_q: Any, cu_seq_lens_k: Any) -> tuple[Any, PackedSequenceInfo | None]:
        """Resolve packed document boundaries against the installed CP chronology."""
        cp_mesh = _context_parallel_mesh(self)
        sequence_info = resolve_packed_sequence(
            {"packed_sequence_info": packed_sequence_info, "cu_seq_lens_q": cu_seq_lens_q,
             "cu_seq_lens_k": cu_seq_lens_k}, hidden_states.shape[1],
            cp_rank=0 if cp_mesh is None else cp_mesh.get_local_rank(),
            cp_size=1 if cp_mesh is None else cp_mesh.size(),
        )
        if sequence_info is not None and hidden_states.shape[0] != 1:
            raise ValueError("K3 packed training requires flattened input_ids [1,T]")
        return cp_mesh, sequence_info

    def _forward_layers(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
                        collect_router_stats: bool, sequence_info: PackedSequenceInfo | None,
                        output_hidden_states: bool) -> tuple[torch.Tensor, Any]:
        """Own immutable block references across decoder and output AttnRes calls."""
        separate_blocks = self.use_attn_residuals and self.config.attn_res_history_layout == "blocks"
        history = (() if separate_blocks else
                   hidden_states.new_empty(hidden_states.shape[0] * hidden_states.shape[1], 0, hidden_states.shape[2]))
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if separate_blocks:
                result = layer(hidden_states, attention_mask, None, collect_router_stats, sequence_info, *history)
            else:
                result = layer(hidden_states, attention_mask, history, collect_router_stats,
                               packed_sequence_info=sequence_info)
            if self.use_attn_residuals:
                if separate_blocks:
                    hidden_states, history = result[0], result[1:]
                else:
                    hidden_states, history = result
            else:
                hidden_states = result
        if self.use_attn_residuals:
            if separate_blocks:
                history = torch.stack(history, dim=1)
            hidden_states = _apply_attn_res(hidden_states.reshape(-1, hidden_states.shape[-1]), history,
                                            self.output_attn_res_proj, self.output_attn_res_norm,
                                            self.config.attn_res_backend).reshape(
                                                hidden_states.shape)
        return hidden_states, all_hidden_states

    def forward(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                past_key_values: Any = None, use_cache: bool | None = None,
                output_hidden_states: bool = False, output_attentions: bool = False,
                return_dict: bool = True, collect_router_stats: bool = True,
                packed_sequence_info: PackedSequenceInfo | None = None,
                cu_seq_lens_q: Any = None, cu_seq_lens_k: Any = None, **kwargs: Any) -> Any:
        """Encode independent rows or explicitly isolated packed documents.

        Args:
            input_ids: Token IDs [batch, local_sequence], exclusive with inputs_embeds.
            inputs_embeds: Precomputed embeddings [batch, local_sequence, hidden_size].
            attention_mask: Optional binary, globally right-padded token validity mask
                in the local [batch, local_sequence] layout.
            position_ids: Optional global contiguous positions or, for packed data,
                document-local positions matching the supplied boundaries.
            past_key_values: Must be None; inference caches are unsupported.
            use_cache: Must be False or None for training.
            output_hidden_states: Include the inputs to decoder layers and the final
                normalized activations in the output history.
            output_attentions: Must be False because KDA exposes no attention maps.
            return_dict: Return BaseModelOutputWithPast when True, otherwise its tuple.
            collect_router_stats: Collect QB observations in this forward; disable
                when replaying activations during backward.
            packed_sequence_info: Validated packed-document metadata, including the
                local CP span; packed input requires batch size one.
            cu_seq_lens_q: Alternative cumulative global query-document offsets.
            cu_seq_lens_k: Matching cumulative global key-document offsets.
            **kwargs: No additional non-None training options are accepted.

        Returns:
            Final normalized activations [batch, local_sequence, hidden_size] and
            optional hidden-state history, with no inference cache or attention maps.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if use_cache or past_key_values is not None:
            raise ValueError("K3 reference model implements training without inference caches")
        if output_attentions:
            raise ValueError("KDA does not expose attention matrices")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        cp_mesh, sequence_info = self._resolve_training_sequence(
            hidden_states, packed_sequence_info, cu_seq_lens_q, cu_seq_lens_k,
        )
        _validate_training_inputs(hidden_states, attention_mask, position_ids, kwargs,
                                  cp_mesh=cp_mesh, sequence_info=sequence_info)
        hidden_states, all_hidden_states = self._forward_layers(
            hidden_states, attention_mask, collect_router_stats, sequence_info, output_hidden_states)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        result = BaseModelOutputWithPast(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
        return result if return_dict else result.to_tuple()


class KimiLinearForCausalLM(KimiPreTrainedModel):
    """K3 causal text head with HF unshifted-label loss semantics."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: KimiLinearConfig) -> None:
        """Create the backbone and vocabulary projection."""
        super().__init__(config)
        self.model = KimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.post_init()

    @classmethod
    def from_config(cls, config: KimiLinearConfig, **kwargs: Any) -> "KimiLinearForCausalLM":
        """Construct through HyperAutoModels without inference-only remote code.

        Args:
            config: Validated K3 text architecture and training backend choices.
            **kwargs: Additional construction options, which are rejected.

        Returns:
            A newly initialized K3 causal language model.
        """
        if kwargs:
            raise ValueError(f"Unsupported K3 LLM construction options: {sorted(kwargs)}")
        return cls(config)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the backbone token embeddings."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the backbone token embeddings.

        Args:
            value: Token embedding module with the configured decoder output width.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Return the vocabulary projection."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Replace the vocabulary projection.

        Args:
            value: Linear head mapping hidden_size to the configured vocabulary.
        """
        self.lm_head = value

    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None, labels: torch.Tensor | None = None,
                shift_labels: torch.Tensor | None = None, valid_token_mask: torch.Tensor | None = None,
                chunk_loss_targets: torch.Tensor | None = None, chunk_loss_mask: torch.Tensor | None = None,
                chunk_loss_chunk_size: int = 1024, chunk_loss_ignore_index: int = -100,
                return_dict: bool = True, **kwargs: Any) -> Any:
        """Train with raw global labels or targets shifted before CP partitioning.

        Chunk targets already encode next-token padding and CP-boundary targets.
        Their mask must never be shifted again on a local sequence shard.

        Args:
            input_ids: Local token IDs [batch, sequence], exclusive with inputs_embeds.
            attention_mask: Local token validity mask for attention and QB.
            inputs_embeds: Precomputed local embeddings [batch, sequence, hidden_size].
            labels: Unshifted HF labels [batch, sequence], supported when no CP
                attention wrappers are installed. The model shifts once and masks
                padding and packed-document transitions.
            shift_labels: Targets shifted globally before CP partitioning, with padding
                and document-crossing targets already masked as -100. These take
                precedence over labels and receive no additional attention-mask filtering.
            valid_token_mask: Alternative to attention_mask, exclusive with it.
            chunk_loss_targets: Globally pre-shifted local targets for chunked
                vocabulary projection, with padding and document crossings already
                masked as -100; exclusive with labels and shift_labels.
            chunk_loss_mask: Optional mask that ignores additional chunk targets.
            chunk_loss_chunk_size: Token chunk size for chunked cross entropy.
            chunk_loss_ignore_index: Must be -100 for the chunked loss path.
            return_dict: Return the named model output when True. Chunked loss
                requires True; the ordinary logits path also supports tuples.
            **kwargs: Backbone options, including packed metadata, position IDs,
                hidden-state collection and router-statistics collection.

        Returns:
            Ordinary execution returns logits and optional mean token loss in
            CausalLMOutputWithPast or its tuple. Chunked execution returns
            ChunkedCausalLMOutput containing loss_sum and valid_token_count without
            materializing full logits; the trainer owns global loss normalization.
        """
        if valid_token_mask is not None:
            if attention_mask is not None:
                raise ValueError("Specify attention_mask or valid_token_mask, not both")
            attention_mask = valid_token_mask
        if chunk_loss_targets is not None and (labels is not None or shift_labels is not None):
            raise ValueError("Specify chunk targets or ordinary labels, not both")
        if labels is not None and shift_labels is None and _context_parallel_mesh(self.model) is not None:
            raise ValueError("CP targets must be shifted globally before sequence partitioning")
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             return_dict=True, **kwargs)
        if chunk_loss_targets is not None:
            return self._chunked_training_output(outputs, chunk_loss_targets, chunk_loss_mask,
                                                 chunk_loss_chunk_size, chunk_loss_ignore_index, return_dict)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if shift_labels is not None:
            if shift_labels.shape != logits.shape[:2]:
                raise ValueError("shift_labels must match the local [batch,sequence] shape")
            total = F.cross_entropy(logits.float().reshape(-1, self.vocab_size), shift_labels.reshape(-1),
                                    ignore_index=-100, reduction="sum")
            loss = total / (shift_labels != -100).sum().clamp_min(1)
        elif labels is not None:
            loss = self._unshifted_label_loss(logits, labels, attention_mask, kwargs)
        result = CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
        return result if return_dict else result.to_tuple()

    def _chunked_training_output(self, outputs: Any, chunk_loss_targets: torch.Tensor,
                                 chunk_loss_mask: torch.Tensor | None, chunk_loss_chunk_size: int,
                                 chunk_loss_ignore_index: int, return_dict: bool) -> ChunkedCausalLMOutput:
        """Keep aligned chunk targets and sum/count normalization in one loss path."""
        if not return_dict or chunk_loss_ignore_index != -100:
            raise ValueError("K3 chunk loss requires return_dict=True and ignore_index=-100")
        if chunk_loss_targets.shape != outputs.last_hidden_state.shape[:2]:
            raise ValueError("chunk_loss_targets must match the local [batch,sequence] shape")
        targets = chunk_loss_targets
        if chunk_loss_mask is not None:
            if chunk_loss_mask.shape != targets.shape:
                raise ValueError("chunk_loss_mask must match chunk targets")
            targets = targets.masked_fill(~chunk_loss_mask.bool(), -100)
        loss_sum = chunked_cross_entropy(outputs.last_hidden_state, targets, self.lm_head.weight,
                                        chunk_size=chunk_loss_chunk_size, ignore_index=-100)
        return ChunkedCausalLMOutput(loss_sum=loss_sum, valid_token_count=targets.ne(-100).sum(),
                                    hidden_states=outputs.hidden_states, logits=None)

    def _unshifted_label_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                              attention_mask: torch.Tensor | None, kwargs: dict[str, Any]) -> torch.Tensor:
        """Shift raw targets once and exclude padding and document transitions."""
        if labels.shape != logits.shape[:2]:
            raise ValueError("labels must have shape [batch, sequence] and be unshifted")
        targets = labels[:, 1:].clone()
        if attention_mask is not None:
            targets.masked_fill_(~attention_mask[:, 1:].bool(), -100)
        sequence_info = resolve_packed_sequence(kwargs, logits.shape[1])
        if sequence_info is not None:
            document_ends = torch.tensor(sequence_info.local_offsets[1:-1], device=targets.device)
            targets[:, document_ends.long() - 1] = -100
        total = F.cross_entropy(logits[:, :-1].float().reshape(-1, self.vocab_size), targets.reshape(-1),
                                ignore_index=-100, reduction="sum")
        return total / (targets != -100).sum().clamp_min(1)
