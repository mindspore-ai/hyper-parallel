# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Standalone Transformer Module - Extracted from Sophon-Pytorch
# This module contains the core Transformer layer/block algorithms with all
# hardware acceleration features preserved, but with all distributed/memory
# optimization logic removed.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_parallel.models.vl_moe.model import (
    VLTextConfig,
)
# from transformers.models.vl_moe.configuration_vl import (
#     VLTextConfig,
# )
from transformers.modeling_layers import GradientCheckpointingLayer

# Import from package modules
from .attention import ATTENTION_CLASSES, _layer_numbers
from .mhc import (
    MhcPreModule,
    MhcPostModule,
    MhcPostProcessModule,
)
from .mlp import TextMLP, LinearWithFusedOps
from .moe import MoELayer

# Conditional NPU imports
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


# ============================================================================
# Utility functions
# ============================================================================

def init_method_normal(sigma):
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_


def scaled_init_method_normal(sigma, num_layers):
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma / math.sqrt(2.0 * num_layers))
    return init_


# ============================================================================
# FusedRMSNorm - Standalone replacement for FusedRMSNorm
# ============================================================================


class FusedRMSNorm(nn.Module):
    """RMS normalization with optional NPU fused kernel."""

    def __init__(self, hidden_size, eps=1e-5, use_fused_rmsnorm=False):
        super().__init__()
        self.eps = eps
        self.use_fused_rmsnorm = use_fused_rmsnorm
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_fused_rmsnorm and HAS_NPU and x.device.type != 'cpu':
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ============================================================================
# Scale - Replacement for Scale module (sandwich post scale)
# ============================================================================

class Scale(nn.Module):
    """Learnable scale factor, used for sandwich post-norm scaling."""

    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x):
        return x * self.weight


# ============================================================================
# bias_dropout_add - Replacement for bias_dropout_add fusion
# ============================================================================

def bias_dropout_add(x_with_bias, residual, prob, training,
                                fp32_residual_connection=False):
    """Bias-dropout-add: add bias (if present), apply dropout, add residual."""
    x, bias = x_with_bias
    if not fp32_residual_connection:
        residual = residual.to(x.dtype)
    if bias is not None:
        x = x + bias
    out = F.dropout(x, p=prob, training=training)
    return residual + out


# ============================================================================
# TextDecoderLayer - Standalone decoder layer
# ============================================================================

class TextDecoderLayer(GradientCheckpointingLayer):
    """Base class for standalone transformer layers.

    Contains the decoder logic (residual connections, normalization,
    and MHC integration) used by regular transformer layers. MTP layers compose
    this module instead of inheriting from it.
    """

    def __init__(self, config: VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_number = layer_idx + 1  # Legacy 1-based internal numbering.
        num_mtp_layers = max(config.mtp_num_total_tokens - 1, 0)
        mtp_start_idx = config.num_hidden_layers - num_mtp_layers
        self.is_mtp = layer_idx >= mtp_start_idx
        self.hidden_dropout = config.hidden_dropout
        self.fp32_residual_connection = config.fp32_residual_connection

        self.use_sandwich_norm = config.sandwich_norm
        self.ffn_pre_norm_scale = config.ffn_pre_norm_scale
        self.attn_pre_norm_scale = config.attn_pre_norm_scale

        # --- Input layernorm ---
        self.input_layernorm = FusedRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            use_fused_rmsnorm=config.use_fused_rmsnorm,
        )
        if config.perform_initialization:
            with torch.no_grad():
                if self.attn_pre_norm_scale != 1.0:
                    self.input_layernorm.weight.data *= self.attn_pre_norm_scale

        # --- Self attention ---
        if self.layer_number in _layer_numbers(config.dsa_layers, offset=1):
            attention_type = "dsa"
        elif config.use_mla:
            attention_type = "mla"
        else:
            attention_type = "gqa"
        self.self_attention = ATTENTION_CLASSES[attention_type](
            config=config,
            layer_number=self.layer_number,
        )

        # --- Sandwich norm for attention ---
        if self.use_sandwich_norm:
            if config.use_sandwich_post_scale:
                self.post_self_attn_layernorm = Scale(
                    dim=config.hidden_size,
                    scale=config.attn_post_norm_scale,
                )
            else:
                self.post_self_attn_layernorm = FusedRMSNorm(
                    hidden_size=config.hidden_size,
                    eps=config.rms_norm_eps,
                    use_fused_rmsnorm=config.use_fused_rmsnorm,
                )
                if config.perform_initialization:
                    with torch.no_grad():
                        scale_val = config.attn_post_norm_scale
                        if scale_val is not None and scale_val != 1.0:
                            self.post_self_attn_layernorm.weight.data *= scale_val

        # --- Pre-MLP layernorm ---
        self.pre_mlp_layernorm = FusedRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            use_fused_rmsnorm=config.use_fused_rmsnorm,
        )
        if config.perform_initialization:
            with torch.no_grad():
                if self.ffn_pre_norm_scale != 1.0:
                    self.pre_mlp_layernorm.weight.data *= self.ffn_pre_norm_scale

        # --- MLP or MoE ---
        if self._should_use_moe():
            self.mlp = MoELayer(config)
        else:
            self.mlp = TextMLP(config)

        # --- Sandwich norm for MLP ---
        if self.use_sandwich_norm:
            if config.use_sandwich_post_scale:
                self.post_mlp_layernorm = Scale(
                    dim=config.hidden_size,
                    scale=config.ffn_post_norm_scale,
                )
            else:
                self.post_mlp_layernorm = FusedRMSNorm(
                    hidden_size=config.hidden_size,
                    eps=config.rms_norm_eps,
                    use_fused_rmsnorm=config.use_fused_rmsnorm,
                )
                if config.perform_initialization:
                    with torch.no_grad():
                        scale_val = self.mlp.get_sandwich_post_norm_scale()
                        if scale_val is not None and scale_val != 1.0:
                            self.post_mlp_layernorm.weight.data *= scale_val

        # --- Post norm ---
        if config.post_norm_layers:
            self.use_post_norm = (self.layer_number - 1) in config.post_norm_layers
        else:
            self.use_post_norm = False

        if self.use_post_norm:
            # When use_mhc and not MTP, hidden_size is expanded by mhc_num_stream
            if config.use_mhc and not self.is_mtp:
                block_ln_hidden_size = config.hidden_size * config.mhc_num_stream
            else:
                block_ln_hidden_size = config.hidden_size
            self.block_post_layernorm = FusedRMSNorm(
                hidden_size=block_ln_hidden_size,
                eps=config.rms_norm_eps,
                use_fused_rmsnorm=config.use_fused_rmsnorm,
            )
            if config.perform_initialization:
                with torch.no_grad():
                    self.block_post_layernorm.weight.data *= config.post_norm_scale
        else:
            self.block_post_layernorm = nn.Identity()

        # --- MHC modules ---
        if config.use_mhc and not self.is_mtp:
            self.attn_mhc_pre_module = MhcPreModule(
                config=config,
                layer_number=self.layer_number,
            )
            self.attn_mhc_post_module = MhcPostModule(config=config)
            self.mlp_mhc_pre_module = MhcPreModule(
                config=config,
                layer_number=self.layer_number,
            )
            self.mlp_mhc_post_module = MhcPostModule(config=config)

        # --- Merge MHC module (conditional) ---
        if config.use_mhc and self.layer_number == (
            config.num_hidden_layers - config.mtp_num_total_tokens + 1
        ) and not self.is_mtp:
            self.merge_mhc_module = MhcPostProcessModule(
                config=config,
                layer_number=self.layer_number,
            )

    def _should_use_moe(self):
        """Determine whether this layer should use MoE instead of dense MLP."""
        if self.config.n_routed_experts is None or self.config.n_routed_experts <= 1:
            return False
        # Layers in moe_layers_replaced_as_dense use dense MLP instead of MoE
        moe_layers_replaced_as_dense = list(range(self.config.first_k_dense_replace))
        if (self.layer_number - 1) in moe_layers_replaced_as_dense:
            return False
        return True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        actual_seq_len=None,
        mome_mask=None,
        kv_reuse_states=None,
    ):
        """Run attention, MLP/MoE, residual, normalization, and optional MHC processing."""
        use_mhc = self.config.use_mhc
        use_mhc_asc = self.config.use_mhc_ascendc_pre
        # ===== Attention block =====
        # MHC pre for attention (or simple residual)
        if not (use_mhc and use_mhc_asc) or self.is_mtp:
            residual = hidden_states
        if use_mhc and not self.is_mtp:
            if use_mhc_asc:
                hidden_states, h_post, h_res, residual = self.attn_mhc_pre_module(
                    hidden_states
                )
            else:
                hidden_states, h_post, h_res, _ = self.attn_mhc_pre_module(
                    hidden_states
                )

        # Input layernorm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention
        attention_output, attention_bias = self.self_attention(
            input_layernorm_output,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            actual_seq_len=actual_seq_len,
            mome_mask=mome_mask,
            kv_reuse_states=kv_reuse_states,
            return_bias=True,
        )
        attention_output_with_bias = attention_output, attention_bias

        # Sandwich norm for attention
        if self.use_sandwich_norm:
            attention_output, attention_bias = attention_output_with_bias
            if attention_bias is not None:
                attention_output = attention_output + attention_bias
            attention_output_with_bias = (
                self.post_self_attn_layernorm(attention_output), None
            )

        # MHC post for attention (or bias_dropout_add)
        if use_mhc and not self.is_mtp:
            hidden_states = self.attn_mhc_post_module(
                attention_output_with_bias[0],
                residual=residual,
                h_post=h_post,
                h_res=h_res,
            )
        else:
            hidden_states = bias_dropout_add(
                attention_output_with_bias,
                residual,
                self.hidden_dropout,
                self.training,
                self.fp32_residual_connection,
            )

        # ===== MLP block =====

        # MHC pre for MLP (or simple residual)
        if not (use_mhc and use_mhc_asc) or self.is_mtp:
            residual = hidden_states

        if use_mhc and not self.is_mtp:
            if use_mhc_asc:
                hidden_states, h_post, h_res, residual = self.mlp_mhc_pre_module(
                    hidden_states
                )
            else:
                hidden_states, h_post, h_res, _ = self.mlp_mhc_pre_module(
                    hidden_states
                )

        # Pre-MLP layernorm
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP / MoE
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # Extract router_logits from MoE layers; dense MLP layers return bias
        router_logits = None
        if self._should_use_moe():
            mlp_output, router_logits = mlp_output_with_bias
            mlp_output_with_bias = (mlp_output, None)

        # Sandwich norm for MLP
        if self.use_sandwich_norm:
            mlp_output, mlp_bias = mlp_output_with_bias
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            mlp_output_with_bias = (
                self.post_mlp_layernorm(mlp_output), None
            )

        # MHC post for MLP (or bias_dropout_add)
        if use_mhc and not self.is_mtp:
            hidden_states = self.mlp_mhc_post_module(
                mlp_output_with_bias[0],
                residual=residual,
                h_post=h_post,
                h_res=h_res,
            )
        else:
            hidden_states = bias_dropout_add(
                mlp_output_with_bias,
                residual,
                self.hidden_dropout,
                self.training,
                self.fp32_residual_connection,
            )

        # Block post layernorm
        hidden_states = self.block_post_layernorm(hidden_states)

        # Merge MHC module (conditional)
        if use_mhc and self.layer_number == (
            self.config.num_hidden_layers - self.config.mtp_num_total_tokens + 1
        ) and not self.is_mtp:
            hidden_states = self.merge_mhc_module(hidden_states)

        # make_viewless_tensor replacement: passthrough
        return hidden_states

# MtpLayer - Dedicated MTP prediction layer
# ============================================================================

class MtpLayer(nn.Module):
    """MTP prediction layer.

    Combines an explicitly provided shifted embedding with the previous hidden
    state, then runs the projected representation through a transformer block.

    The transformer block is constructed before the MTP-specific modules to
    preserve the parameter initialization order of the former inheritance-based
    implementation.
    """

    def __init__(self, config: VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_number = layer_idx + 1

        # Keep this first: the inheritance-based implementation initialized the
        # transformer modules before creating the MTP projection modules.
        self.mtp_block = TextDecoderLayer(
            config=config,
            layer_idx=layer_idx,
        )
        # MTP layers do not apply the optional block-level post norm.
        self.mtp_block.block_post_layernorm = nn.Identity()

        # MTP-specific modules
        self.prev_norm = FusedRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            use_fused_rmsnorm=config.use_fused_rmsnorm,
        )
        self.emb_norm = FusedRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            use_fused_rmsnorm=config.use_fused_rmsnorm,
        )
        self.prev_proj = LinearWithFusedOps(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.prev_proj._init_role = "input"
        if config.perform_initialization:
            config._standalone_init_weights(self.prev_proj)

    def forward(
        self,
        inputs_embeds,
        previous_hidden_state,
        attention_mask=None,
        position_embeddings=None,
        actual_seq_len=None,
        mome_mask=None,
        kv_reuse_states=None,
    ):
        mtp_input = self.prev_proj(
            torch.cat(
                [
                    self.prev_norm(previous_hidden_state),
                    self.emb_norm(inputs_embeds),
                ],
                dim=-1,
            )
        )[0]

        # Forward through the composed transformer block.
        output = self.mtp_block(
            mtp_input,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            actual_seq_len=actual_seq_len,
            mome_mask=mome_mask,
            kv_reuse_states=kv_reuse_states,
        )

        return output