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
"""VL-MoE model for conditional generation.

Migrated from LlamaFactory/third_party/transformers/src/transformers/models/vl_moe/.
This module is self-contained — it imports from the standard ``transformers`` package,
and from ``utils`` (module, embedding, mask_calculation).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make utils packages importable.
# utils lives at LlamaFactory/third_party/utils/ and exposes
#   - module  (layers, MoE, MHC, attention)
#   - embedding           (MultimodalRotaryEmbedding)
#   - mask_calculation    (calculate_masks)
# ---------------------------------------------------------------------------
_utils_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "LlamaFactory", "third_party", "utils",
)
_utils_DIR = os.path.realpath(_utils_DIR)
if _utils_DIR not in sys.path:
    sys.path.insert(0, _utils_DIR)

# ---------------------------------------------------------------------------
# Standard transformers imports (NOT from LlamaFactory's fork)
# ---------------------------------------------------------------------------
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast, ModelOutput
from transformers.processing_utils import Unpack

from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import TransformersKwargs, can_return_tuple, logging
from transformers.utils.generic import OutputRecorder, check_model_inputs

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.activations import ACT2FN
from transformers.utils import is_torch_npu_available

# ---------------------------------------------------------------------------
# utils imports are deferred until *after* the config classes below are
# defined. utils's ``module.layer`` imports
# ``VLTextConfig`` back from this module at top level (it only needs
# it as a stringified type annotation under ``from __future__ import
# annotations``). Importing utils here -- before the config classes exist
# -- triggers a circular import. By moving these imports below the config
# classes, the back-import from ``layer.py`` resolves successfully.
# ---------------------------------------------------------------------------

# NPU attention support
if is_torch_npu_available() and "910" in torch.npu.get_device_name():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    NPU_ATTN_INFR = True
else:
    NPU_ATTN_INFR = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

logger = logging.get_logger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

class VLVisionConfig(PretrainedConfig):
    r"""
    Configuration class to store the configuration of the vision backbone.
    """

    model_type = "vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        num_heads=16,
        hidden_size=1024,
        intermediate_size=4096,
        hidden_act="gelu",
        layer_norm_type="LayerNorm",
        rms_norm_eps=1e-06,
        use_norm_pre=True,
        norm_pre_eps=1e-05,
        initializer_range=0.02,
        rope_theta=10000.0,
        rope_scaling=None,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        in_chans=None,
        spatial_merge_size=2,
        out_hidden_size=3584,
        tokens_per_second=2,
        window_size=112,
        fullatt_block_indexes=None,
        mm_unit_vision_select_layer=None,
        use_gatedmerger=True,
        position_embedding_type="3d_rope",
        mrope_section="8,12,12",
        rotary_interleaved=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_type = layer_norm_type
        self.rms_norm_eps = rms_norm_eps
        self.use_norm_pre = use_norm_pre
        self.norm_pre_eps = norm_pre_eps
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.position_embedding_type = position_embedding_type
        self.mrope_section = mrope_section
        self.rotary_interleaved = rotary_interleaved
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels if in_chans is None else in_chans
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes if fullatt_block_indexes is not None else [i for i in range(self.depth)]
        self.mm_unit_vision_select_layer = mm_unit_vision_select_layer if mm_unit_vision_select_layer is not None else [-1, -3]
        self.use_gatedmerger = use_gatedmerger


class VLTextConfig(PretrainedConfig):
    r"""
    Configuration class to store the configuration of the text backbone.
    """

    model_type = "vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def get_text_config(self, decoder=None, encoder=None):
        """This config *is* the text config — return self instead of searching
        for a nested ``text_config`` attribute (which may be a plain dict
        injected by PretrainedConfig's kwargs passthrough, causing
        ``'dict' object has no attribute 'to_dict'`` in
        ``GenerationConfig.from_model_config``)."""
        return self

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=46,
        num_attention_heads=48,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=6400000.0,
        rope_scaling=None,
        rope_interleaved=False,
        attention_dropout=0.0,
        attention_bias=False,
        # MLA parameters
        use_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_mm_split=False,
        padded_base_length=1,
        # MoE parameters
        first_k_dense_replace=2,
        moe_intermediate_size=1024,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        router_enable_expert_bias=True,
        router_sliding_window=3,
        n_group=None,
        topk_group=1,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_aux_level="microbatch",
        moe_grouped_gemm=True,
        moe_use_sigmoid_gating=True,
        moe_router_pre_softmax=False,
        moe_input_jitter_eps=None,
        moe_z_loss_coeff=None,
        router_gating_in_fp32=False,
        enable_routing_replay=False,
        experts_2d=False,
        moe_layer_recompute=False,
        use_fused_permute=False,
        moe_router_scale=False,
        # MHC parameters
        use_mhc=True,
        mhc_num_stream=4,
        mhc_use_gamma=True,
        mhc_recur_norm=20,
        hc_eps=1e-6,
        mhc_init_alpha=0.01,
        mhc_init_gamma=1.0,
        mhc_hpre_renorm=False,
        mhc_expand_linear=False,
        use_mhc_ascendc_pre=True,
        use_mhc_ascendc_post=True,
        # Norm parameters
        sandwich_norm=True,
        use_sandwich_post_scale=False,
        attn_pre_norm_scale=1.0,
        ffn_pre_norm_scale=1.0,
        attn_post_norm_scale=1.0,
        moe_attn_post_norm_scale=1.0,
        ffn_post_norm_scale=1.0,
        post_norm_layers=None,
        block_post_layernorm_idx=None,
        post_norm_scale=1.0,
        # Param sink parameters
        param_sink_number=128,
        param_sink_with_value=False,
        param_sink_scalar=None,
        param_sink_of_head_num=False,
        use_fused_sink_fa=True,
        apply_FA_rescale=False,
        # DSA parameters
        dsa_layers=None,
        dsa_loss_coeff=0.0,
        dsa_dense_warm_up=False,
        freeze_DSA=False,
        index_head_dim=256,
        index_num_attention_heads=1,
        index_topk=64,
        dense_warmup_chunk_size=1,
        use_fused_rotary_pos_emb=True,
        # MTP parameters
        num_nextn_predict_layers=3,
        mtp_num_total_tokens=None,
        mtp_apply_separate_embedding=False,
        mtp_loss_weight=1.0,
        # MoME parameters
        use_mome=True,
        use_fused_mome=False,
        # SWA parameters
        use_sliding_window=False,
        sliding_window_list=None,
        swa_layers=None,
        swa_attention_sink=0,
        # Other parameters
        layer_types=None,
        add_qkv_bias=None,
        add_dense_bias=None,
        attn_groupnorm=False,
        attn_elementwise_gate=False,
        use_flash_attn=True,
        gqa_qkv_clone_for_mem=False,
        kv_reuse_mapping=None,
        # Standalone transformer internal parameters
        normalization="RMSNorm",
        use_fused_rmsnorm=True,
        perform_initialization=True,
        gated_linear_unit=True,
        use_fused_swiglu=False,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        fp32_residual_connection=False,
        mask_type="causal",
        reset_attention_mask=True,
        mask_compress=True,
        micro_batch_size=1,
        emb_init_method_std=0.0125,
        moe_init_method_std=0.00206,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleaved = rope_interleaved
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        # MLA
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_mm_split = mla_mm_split
        self.padded_base_length = padded_base_length

        # MoE
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.router_enable_expert_bias = router_enable_expert_bias
        self.router_sliding_window = router_sliding_window
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_router_load_balancing_type = moe_router_load_balancing_type
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_aux_level = moe_aux_level
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_use_sigmoid_gating = moe_use_sigmoid_gating
        self.moe_router_pre_softmax = moe_router_pre_softmax
        self.moe_input_jitter_eps = moe_input_jitter_eps
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.router_gating_in_fp32 = router_gating_in_fp32
        self.enable_routing_replay = enable_routing_replay
        self.experts_2d = experts_2d
        self.moe_layer_recompute = moe_layer_recompute
        self.use_fused_permute = use_fused_permute
        self.moe_router_scale = moe_router_scale

        # MHC
        self.use_mhc = use_mhc
        self.mhc_num_stream = mhc_num_stream
        self.mhc_use_gamma = mhc_use_gamma
        self.mhc_recur_norm = mhc_recur_norm
        self.hc_eps = hc_eps
        self.mhc_init_alpha = mhc_init_alpha
        self.mhc_init_gamma = mhc_init_gamma
        self.mhc_hpre_renorm = mhc_hpre_renorm
        self.mhc_expand_linear = mhc_expand_linear
        self.use_mhc_ascendc_pre = use_mhc_ascendc_pre
        self.use_mhc_ascendc_post = use_mhc_ascendc_post

        # Norm
        self.sandwich_norm = sandwich_norm
        self.use_sandwich_post_scale = use_sandwich_post_scale
        self.attn_pre_norm_scale = attn_pre_norm_scale
        self.ffn_pre_norm_scale = ffn_pre_norm_scale
        self.attn_post_norm_scale = attn_post_norm_scale
        self.moe_attn_post_norm_scale = moe_attn_post_norm_scale
        self.ffn_post_norm_scale = ffn_post_norm_scale
        self.post_norm_layers = post_norm_layers if post_norm_layers is not None else block_post_layernorm_idx
        self.post_norm_scale = post_norm_scale

        # Param sink
        self.param_sink_number = param_sink_number
        self.param_sink_with_value = param_sink_with_value
        self.param_sink_scalar = param_sink_scalar
        self.param_sink_of_head_num = param_sink_of_head_num
        self.use_fused_sink_fa = use_fused_sink_fa
        self.apply_FA_rescale = apply_FA_rescale

        # DSA
        self.dsa_layers = dsa_layers
        self.dsa_loss_coeff = dsa_loss_coeff
        self.dsa_dense_warm_up = dsa_dense_warm_up
        self.freeze_DSA = freeze_DSA
        self.index_head_dim = index_head_dim
        self.index_num_attention_heads = index_num_attention_heads
        self.index_topk = index_topk
        self.dense_warmup_chunk_size = dense_warmup_chunk_size
        self.use_fused_rotary_pos_emb = use_fused_rotary_pos_emb

        # MTP
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_num_total_tokens = (
            num_nextn_predict_layers if mtp_num_total_tokens is None else mtp_num_total_tokens
        )
        self.mtp_apply_separate_embedding = mtp_apply_separate_embedding
        self.mtp_loss_weight = mtp_loss_weight

        # MoME
        self.use_mome = use_mome
        self.use_fused_mome = use_fused_mome

        # SWA
        self.use_sliding_window = use_sliding_window
        self.sliding_window_list = sliding_window_list
        self.swa_layers = swa_layers
        self.swa_attention_sink = swa_attention_sink

        # Other
        self.layer_types = layer_types
        self.add_qkv_bias = add_qkv_bias if add_qkv_bias is not None else attention_bias
        self.add_dense_bias = add_dense_bias if add_dense_bias is not None else attention_bias
        self.attn_groupnorm = attn_groupnorm
        self.attn_elementwise_gate = attn_elementwise_gate
        self.use_flash_attn = use_flash_attn
        self.gqa_qkv_clone_for_mem = gqa_qkv_clone_for_mem
        self.kv_reuse_mapping = kv_reuse_mapping

        # Standalone transformer internal
        self.normalization = normalization
        self.use_fused_rmsnorm = use_fused_rmsnorm
        self.perform_initialization = perform_initialization
        self.gated_linear_unit = gated_linear_unit
        self.use_fused_swiglu = use_fused_swiglu
        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.fp32_residual_connection = fp32_residual_connection
        self.mask_type = mask_type
        self.reset_attention_mask = reset_attention_mask
        self.mask_compress = mask_compress
        self.micro_batch_size = micro_batch_size
        self.emb_init_method_std = emb_init_method_std
        self.moe_init_method_std = moe_init_method_std

        # Derive layer_types from swa_layers if not provided
        if layer_types is None and swa_layers is not None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window_list is not None and i in swa_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # RoPE scaling compatibility
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        rope_config_validation(self, ignore_keys={"mrope_section", "mrope_interleaved", "rotary_mode"})

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class VLConfig(PretrainedConfig):
    r"""
    Configuration class to store the configuration of the full VL-MoE model.
    """

    model_type = "vl_moe"
    sub_configs = {"vision_config": VLVisionConfig, "text_config": VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=148909,
        video_token_id=148910,
        vision_start_token_id=148907,
        vision_end_token_id=148908,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

    def __getattr__(self, key):
        """Delegate attribute lookups to ``text_config`` when the key is not
        found on the composite config itself.

        This allows code that was written for single-config models (e.g.
        ``config.vocab_size``, ``config.num_hidden_layers``) to work
        transparently with the composite VL config without every caller
        needing to know about ``config.text_config.xxx``.
        """
        # Avoid infinite recursion during __init__ before sub-configs are set
        text_cfg = self.__dict__.get("text_config")
        if text_cfg is not None and hasattr(text_cfg, key):
            return getattr(text_cfg, key)
        # Fall back to the default AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


# ===========================================================================
# utils imports
# ---------------------------------------------------------------------------
# Deferred until here (after the config classes above are defined) to break a
# circular import: ``module.layer`` imports
# ``VLTextConfig`` back from this module at top level. With the
# config classes already defined, that back-import now succeeds. These symbols
# are required at class-definition time by ``PreTrainedModel`` below
# (e.g. in ``_no_split_modules`` / ``_can_record_outputs``), so they must be
# imported before that class body is evaluated.
# ===========================================================================
from module import (
    MtpLayer,
    FusedRMSNorm,
    TextDecoderLayer,
)
from module.moe import Experts, MoELayer, SharedExpert
from module.mhc import MhcPreModule, MhcPostModule, MhcPostProcessModule
from embedding import MultimodalRotaryEmbedding
from mask_calculation import calculate_masks


# ===========================================================================
# Text model (from modeling.py)
# ===========================================================================

class mHCModule(nn.Module):
    """Wrapper that delegates to standalone MHC modules."""

    def __init__(self, config: VLTextConfig, merge_layer_only_pre=False):
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.merge_layer_only_pre = merge_layer_only_pre

        if not self.merge_layer_only_pre:
            self.mhc_pre_module = MhcPreModule(config, layer_number=1)
            self.mhc_post_module = MhcPostModule(config)
        else:
            self.mhc_pre_module = MhcPostProcessModule(config, layer_number=1)

    def hc_pre(self, x: torch.Tensor):
        if not self.merge_layer_only_pre:
            y, h_post, h_res, _residual = self.mhc_pre_module(x)
            return y, h_post, h_res
        else:
            y = self.mhc_pre_module(x)
            return y, None, None

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor):
        if self.merge_layer_only_pre:
            return x
        return self.mhc_post_module(x, residual, h_post, h_res)


@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PreTrainedModel(PreTrainedModel):
    config: VLTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TextDecoderLayer", "MtpLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(MoELayer, index=1),
        "hidden_states": TextDecoderLayer,
    }

    def _init_weights(self, module, init_role=None):
        init_role = init_role or getattr(module, "_init_role", None)
        if getattr(module, "_embedding_init", False):
            std = self.config.hidden_size ** -0.5
            torch.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
            return
        if isinstance(module, nn.Parameter):
            std = self.config.initializer_range
            if init_role == "output":
                std /= math.sqrt(2.0 * self.config.num_hidden_layers)
            elif init_role != "input":
                raise ValueError(f"Unknown initialization role: {init_role}")
            torch.nn.init.normal_(module, mean=0.0, std=std)
            return
        if (
            isinstance(module, nn.Linear)
            and init_role is not None
            and self.config.perform_initialization
            and not getattr(module, "_role_init_applied", False)
        ):
            std = self.config.initializer_range
            if init_role == "output":
                std /= math.sqrt(2.0 * self.config.num_hidden_layers)
            elif init_role != "input":
                raise ValueError(f"Unknown initialization role: {init_role}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            module._role_init_applied = True
        elif isinstance(module, MoELayer):
            module.gate.weight.data.normal_(mean=0.0, std=self.config.moe_init_method_std)
        elif isinstance(module, Experts):
            module.gate_up_proj.data.normal_(mean=0.0, std=self.config.moe_init_method_std)
            module.down_proj.data.normal_(
                mean=0.0,
                std=self.config.moe_init_method_std / math.sqrt(2.0 * self.config.num_hidden_layers),
            )
            if module.add_bias:
                module.bias1.data.zero_()
                module.bias2.data.zero_()
        elif isinstance(module, SharedExpert):
            module.linear_fc1.weight.data.normal_(mean=0.0, std=self.config.moe_init_method_std)
            module.linear_fc2.weight.data.normal_(
                mean=0.0,
                std=self.config.moe_init_method_std / math.sqrt(2.0 * self.config.num_hidden_layers),
            )
        else:
            super()._init_weights(module)


class Model(PreTrainedModel):
    def __init__(self, config: VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        config._attn_implementation = "eager"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_tokens._embedding_init = True
        _mrope_section = None
        if config.rope_scaling and "mrope_section" in config.rope_scaling:
            _mrope_section = config.rope_scaling["mrope_section"]
        if _mrope_section is None:
            _mrope_section = [8, 12, 12]

        self.rotary_emb = MultimodalRotaryEmbedding(kv_channels=config.qk_rope_head_dim,
                                                    rotary_interleaved=config.rope_interleaved,
                                                    qk_rope_head_dim=config.qk_rope_head_dim,
                                                    rotary_base=config.rope_theta,
                                                    mrope_section=_mrope_section,
                                                    mrope_ids_interleaved=True,
                                                    )
        self._init_weights(self.embed_tokens)
        del self.embed_tokens._embedding_init
        self.num_mtp_layers = max(config.mtp_num_total_tokens - 1, 0)
        config._standalone_init_weights = self._init_weights
        try:
            self.layers = nn.ModuleList(self.build_layers())
        finally:
            del config._standalone_init_weights
        if config.mtp_num_total_tokens > 1:
            self.final_layernorms = nn.ModuleList(
                [
                    FusedRMSNorm(
                        hidden_size=config.hidden_size,
                        eps=config.rms_norm_eps,
                        use_fused_rmsnorm=config.use_fused_rmsnorm,
                    )
                    for _ in range(config.mtp_num_total_tokens)
                ]
            )
        else:
            self.final_layernorm = FusedRMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                use_fused_rmsnorm=config.use_fused_rmsnorm,
            )
        self.gradient_checkpointing = False
        self.has_sliding_layers = self.config.layer_types is not None \
                                  and "sliding_attention" in self.config.layer_types
        self.use_mhc = config.use_mhc
        if self.use_mhc:
            self.mhc_num_stream = config.mhc_num_stream
        self.post_init()

    def build_layers(self):
        mtp_start_idx = self.config.num_hidden_layers - self.num_mtp_layers
        return [
            (
                TextDecoderLayer(
                    config=self.config,
                    layer_idx=layer_idx,
                )
                if layer_idx < mtp_start_idx
                else MtpLayer(self.config, layer_idx=layer_idx)
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ]

    @check_model_inputs
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            mome_mask: torch.Tensor | None = None,
            swa_mask: torch.Tensor | None = None,
            actual_seq_len=None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            cache_position: torch.LongTensor | None = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.use_mhc:
                inputs_embeds = torch.cat([inputs_embeds] * self.mhc_num_stream, dim=-1)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(position_ids)

        mtp_active = self.config.mtp_num_total_tokens > 1
        mtp_inputs = []
        if mtp_active:
            if self.config.mtp_apply_separate_embedding:
                raise NotImplementedError(
                    "Explicit MTP inputs for mtp_apply_separate_embedding=True are not supported yet."
                )
            seq_length = hidden_states.size(1)
            single_stream = hidden_states[..., : self.config.hidden_size]
            padded = torch.nn.functional.pad(
                single_stream,
                (0, 0, 0, self.config.mtp_num_total_tokens - 1),
                value=0,
            )
            mtp_inputs = [
                padded[:, offset:offset + seq_length]
                for offset in range(1, self.config.mtp_num_total_tokens)
            ]

        mtp_start_idx = len(self.layers) - self.num_mtp_layers
        mtp_hidden_states = []
        previous_hidden_state = None
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < mtp_start_idx:
                hidden_states = layer(
                    hidden_states.clone() if mtp_active else hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    actual_seq_len=actual_seq_len,
                    mome_mask=mome_mask,
                )
                continue

            if previous_hidden_state is None:
                previous_hidden_state = hidden_states
                mtp_hidden_states.append(hidden_states)
            mtp_layer_idx = layer_idx - mtp_start_idx
            previous_hidden_state = layer(
                inputs_embeds=mtp_inputs[mtp_layer_idx],
                previous_hidden_state=previous_hidden_state,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                actual_seq_len=actual_seq_len,
                mome_mask=mome_mask,
            )
            mtp_hidden_states.append(previous_hidden_state)

        if mtp_active:
            mtp_hidden_states = [
                layernorm(mtp_hidden_state)
                for layernorm, mtp_hidden_state in zip(self.final_layernorms, mtp_hidden_states)
            ]
            hidden_states = torch.concat(mtp_hidden_states, dim=2)
        else:
            hidden_states = self.final_layernorm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class ForCausalLM(PreTrainedModel, GenerationMixin):
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            cache_position: torch.LongTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


# ===========================================================================
# Vision-Language model
# ===========================================================================

def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor


class EmbeddedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        if self.hidden_act == "silu":
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        if self.hidden_act == "silu":
            x_gate = self.gate_proj(hidden_state)
            x_gate = self.act_fn(x_gate)
            x_up = self.up_proj(hidden_state)
            intermediate_parallel = x_gate * x_up
        else:
            x_up = self.up_proj(hidden_state)
            intermediate_parallel = self.act_fn(x_up)
        x_down = self.down_proj(intermediate_parallel)
        return x_down


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2,
                 in_channels: int = 3, embed_dim: int = 1152) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
        self.input_size = self.patch_size * self.patch_size * in_channels * self.temporal_patch_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.input_size:
            hidden_states = torch.cat([hidden_states.reshape(-1, self.patch_size * self.patch_size),
                                       hidden_states.reshape(-1, self.patch_size * self.patch_size)], dim=-1).reshape(
                -1, self.input_size)
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class VLPatchEmbed(VisionPatchEmbed):
    pass


class VideoRotaryEmbedding3D(nn.Module):
    """3D (T,H,W) RoPE frequency generator."""

    def __init__(self, kv_channels: int, rotary_base: float = 10000.0,
                 mrope_section: Optional[list[int]] = None):
        super().__init__()
        head_dim = kv_channels
        half = head_dim // 2
        if mrope_section is not None:
            self.t_size, self.h_size, self.w_size = mrope_section
        else:
            unit = half // 16
            self.t_size = 4 * unit
            self.h_size = 6 * unit
            self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (rotary_base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (rotary_base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (rotary_base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, t: int, h: int, w: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.inv_freq_t.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)
        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)
        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)
        return torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)
        t_pos = patch_positions[..., 0].float()
        h_pos = patch_positions[..., 1].float()
        w_pos = patch_positions[..., 2].float()
        ft = torch.einsum("bs,d->bsd", t_pos, inv_t)
        fh = torch.einsum("bs,d->bsd", h_pos, inv_h)
        fw = torch.einsum("bs,d->bsd", w_pos, inv_w)
        return torch.cat([ft, fh, fw], dim=-1)


class VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class RMSNorm(EmbeddedRMSNorm):
    pass


class VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, use_gatedmerger: bool = False) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.gated = use_gatedmerger
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.gate_act = nn.SiLU() if use_gatedmerger else None
        outdim = dim * 2 if use_gatedmerger else dim
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, outdim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.size()
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        if self.gated:
            x, gate = torch.chunk(x, 2, dim=-1)
            x = x * self.gate_act(gate)
        x = x.view(b, s // (self.spatial_merge_size ** 2), -1)
        return x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply_rotary_pos_emb_npu(t: torch.Tensor, freqs: torch.Tensor,
                             rotary_interleaved: bool = False) -> torch.Tensor:
    rot_dim = freqs.shape[-1]
    t_dim = t.shape[-1]
    if rot_dim != t_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    else:
        t_pass = None
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    rotary_mode = 'interleave' if rotary_interleaved else 'half'
    t_copy = t.clone()
    t = torch_npu.npu_rotary_mul(t_copy, cos_, sin_, rotary_mode=rotary_mode)
    t_copy.untyped_storage().resize_(0)
    if t_pass is not None:
        return torch.cat((t, t_pass), dim=-1)
    return t


class VLVisionAttention(nn.Module):
    def __init__(self, config: VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False
        self.rotary_interleaved = config.rotary_interleaved

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, attention_mask=None, **kwargs):
        seq_length = hidden_states.shape[0]
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.view(seq_length, self.num_heads, 3 * self.head_dim)
        query_states, key_states, value_states = torch.split(mixed_qkv, [self.head_dim, self.head_dim, self.head_dim], dim=-1)

        q_pos_emb, k_pos_emb = position_embeddings
        query_states = apply_rotary_pos_emb_npu(query_states, q_pos_emb.squeeze(1), self.rotary_interleaved)
        key_states = apply_rotary_pos_emb_npu(key_states, k_pos_emb.squeeze(1), self.rotary_interleaved)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if not self.training and NPU_ATTN_INFR:
            if isinstance(cu_seqlens, torch.Tensor):
                cu_seqlens = cu_seqlens.tolist()
            q, k, v = [rearrange(x, "b n s d -> (b s) n d") for x in [query_states, key_states, value_states]]
            attn_output = torch_npu.npu_fusion_attention(
                q, k, v, self.num_heads, "TND",
                pse=None, padding_mask=None, atten_mask=None,
                scale=self.scaling, pre_tockens=1048576, next_tockens=0,
                keep_prob=1.0, inner_precise=0, sparse_mode=0,
                actual_seq_qlen=cu_seqlens, actual_seq_kvlen=cu_seqlens,
            )[0]
        else:
            attn_output, _ = attention_interface(
                self, query_states, key_states, value_states,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen, max_length_k=max_seqlen,
                is_causal=False, **kwargs,
            )

        attn_output = attn_output.reshape(seq_length, 1, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.pre_mlp_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = VLVisionAttention(config=config)
        self.mlp = VLMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, attention_mask=None, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.input_layernorm(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask, **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.pre_mlp_layernorm(hidden_states))
        return hidden_states


class PreTrainedModel(PreTrainedModel):
    config_class = VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TextDecoderLayer", "MtpLayer", "VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            copy_(module.inv_freq, inv_freq)


class VisionTransformerPretrainedModel(PreTrainedModel):
    config_class = VLVisionConfig
    _no_split_modules = ["VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = VLPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        self.layernorm_pre = nn.LayerNorm(config.hidden_size, eps=1e-5)
        head_dim = config.hidden_size // config.num_heads
        self.position_embedding_type = config.position_embedding_type
        mrope_section_str = config.mrope_section
        mrope_section = [int(x) for x in mrope_section_str.split(',')] if isinstance(mrope_section_str, str) else mrope_section_str
        rope_theta = config.rope_theta

        if self.position_embedding_type == '3d_rope':
            self.rotary_pos_emb = VideoRotaryEmbedding3D(
                kv_channels=head_dim, rotary_base=rope_theta, mrope_section=mrope_section,
            )
        else:
            self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, theta=rope_theta)

        self.blocks = nn.ModuleList([VLVisionBlock(config) for _ in range(config.depth)])
        self.select_layer = config.mm_unit_vision_select_layer
        self.select_index = [config.depth + i for i in self.select_layer]
        self.select_index = self.select_index[::-1]
        self.select_layer = [-1 * (i + 1) for i in range(len(self.select_index))]
        self.use_gatedmerger = config.use_gatedmerger
        if config.use_gatedmerger:
            self.merger = VLPatchMerger(
                dim=config.out_hidden_size, context_dim=config.hidden_size,
                spatial_merge_size=config.spatial_merge_size, use_gatedmerger=True,
            )
        else:
            self.merger = nn.ModuleList([
                VLPatchMerger(
                    dim=config.out_hidden_size, context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                )
                for i in range(len(self.select_layer))
            ])
        self.gradient_checkpointing = False
        self.take_indices = self.select_index

    def _compute_pos_ids_3d(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t_val, h, w in grid_thw:
            t_val, h, w = int(t_val), int(h), int(w)
            ms = self.spatial_merge_size
            tpos_ids = torch.arange(t_val).repeat_interleave(h * w)
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h // ms, ms, w // ms, ms).permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h // ms, ms, w // ms, ms).permute(0, 2, 1, 3).flatten()
            hw_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t_val, 1)
            thw_ids = torch.cat([tpos_ids.unsqueeze(-1), hw_ids], dim=-1)
            pos_ids.append(thw_ids)
        return torch.cat(pos_ids, dim=0)

    def _prepare_attention_mask(self, inputs_tensor, cu_seqlens):
        if self.config._attn_implementation == "flash_attention_2":
            return None
        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length], torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device, dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0
        return attention_mask

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.layernorm_pre(hidden_states)
        hidden_states = hidden_states.unsqueeze(1)

        device = hidden_states.device
        if self.position_embedding_type == '3d_rope':
            rot_pos_ids = self._compute_pos_ids_3d(grid_thw).to(device)
            rotary_pos_emb = self.rotary_pos_emb.forward_from_positions(
                rot_pos_ids.unsqueeze(0)
            ).squeeze(0)
        else:
            rotary_pos_emb = self.rot_pos_emb(grid_thw)

        if self.position_embedding_type == '3d_rope':
            rotary_pos_emb = rotary_pos_emb.to(device)
            rotary_pos_emb = rotary_pos_emb.unsqueeze(1)
            rotary_pos_emb = rotary_pos_emb.repeat(1, 1, 2).unsqueeze(2)
            position_embeddings = (rotary_pos_emb, rotary_pos_emb)
        else:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        intermediates = []
        for layer_num, blk in enumerate(self.blocks):
            attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask, **kwargs,
            )
            if layer_num in self.take_indices:
                intermediates.append(hidden_states)

        hidden_states = hidden_states.transpose(0, 1)
        if self.use_gatedmerger:
            hidden_states = self.merger(hidden_states)
        else:
            image_embeddings_list = []
            for idx, sl in enumerate(self.select_layer):
                image_embeddings_list.append(self.merger[idx](intermediates[sl].transpose(0, 1)))
            hidden_states = sum(image_embeddings_list)
        return hidden_states


@dataclass
class VLModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None


class VLRotaryEmbedding(nn.Module):
    def __init__(self, config: VLTextConfig, device=None):
        super().__init__()
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            self.mrope_interleaved = config.rope_scaling.get("mrope_interleaved", False)
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        mrope_section = config.rope_scaling.get("mrope_section", None) if config.rope_scaling else None
        self.mrope_section = mrope_section
        if self.mrope_interleaved:
            if not self.mrope_section:
                raise AssertionError("when you use interleave mrope, mrope_section cannot be None.")
            if len(mrope_section) == 2:
                h_num, w_num = mrope_section[0], mrope_section[1]
                mrope_dim = self.get_mrope_interleaved_id_list(h_num, w_num, 0)
            elif len(mrope_section) == 3:
                t_num, h_num, w_num = mrope_section[0], mrope_section[1], mrope_section[2]
                mrope_dim = self.get_mrope_interleaved_id_list(t_num, h_num, w_num, force_last=True)
            else:
                raise AssertionError("Cannot support the length of mrope section is not 2 or 3.")
            mrope_dim = mrope_dim * 2
            self.mrope_dim = mrope_dim

    @staticmethod
    def compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
        if config is not None and len(rope_kwargs) > 0:
            raise ValueError("Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive")
        if len(rope_kwargs) > 0:
            base = rope_kwargs["base"]
            dim = rope_kwargs["dim"]
        elif config is not None:
            base = config.rope_theta
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = config.head_dim
            dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            if self.mrope_interleaved:
                mrope_section_3d = [1] * len(self.mrope_dim)
                mrope_dim = self.mrope_dim
                emb = torch.cat([m[mrope_dim[i]] for i, m in enumerate(emb.split(mrope_section_3d, dim=-1))], dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            if not self.mrope_interleaved and self.mrope_section:
                mrope_section = self.mrope_section * 2
                cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
                sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def get_mrope_interleaved_id_list(a: int, b: int, c: int, force_last: bool = False) -> list[int]:
        if force_last:
            a -= 1
        counts = {0: a, 1: b, 2: c}
        placed = dict.fromkeys(counts, 0)
        rem = counts.copy()
        seq: list[int] = []
        last = None
        total = a + b + c
        for _ in range(total):
            cands = [k for k in rem if rem[k] > 0 and k != last]
            if not cands:
                cands = [k for k in rem if rem[k] > 0]
            try:
                best = min(cands, key=lambda k: (placed[k] / counts[k], k))
            except KeyError:
                best = 0
            seq.append(best)
            placed[best] += 1
            rem[best] -= 1
            last = best
        if force_last:
            seq.append(0)
        return seq


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class KwargsForCausalLM(FlashAttentionKwargs): ...


class ProjectionSingle(nn.Module):
    def __init__(self, i_hidden_size: int, t_hidden_size: int):
        super().__init__()
        self.act = F.gelu
        self.fc1 = nn.Linear(i_hidden_size, t_hidden_size, bias=True)

    def forward(self, hidden_states):
        x = self.act(hidden_states)
        return self.fc1(x)


class VLTextModel(Model):
    def __init__(self, config: VLTextConfig):
        super().__init__(config)


class VLModel(PreTrainedModel):
    base_model_prefix = ""
    config_class = VLConfig
    _no_split_modules = ["TextDecoderLayer", "MtpLayer", "VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = VLTextModel(config.text_config)
        self.rope_deltas = None
        self.use_mhc = self.config.text_config.use_mhc
        self.mhc_num_stream = self.config.text_config.mhc_num_stream
        self.visual.vision_projection = ProjectionSingle(config.vision_config.out_hidden_size,
                                                         config.text_config.hidden_size)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(self, input_ids=None, image_grid_thw=None, video_grid_thw=None,
                       second_per_grid_ts=None, attention_mask=None):
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        tokens_per_second = self.config.vision_config.tokens_per_second
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1],
                                      dtype=input_ids.dtype, device=input_ids.device)
            video_idx = 0
            image_idx = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                input_tokens = input_ids.tolist()
                src_item = input_tokens
                new_src_item: list[int] = []
                llm_pos_ids_list: list[torch.Tensor] = []
                idx = 0
                while idx < len(src_item):
                    new_src_item_len = len(new_src_item)
                    start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if src_item[idx] not in [video_token_id, image_token_id]:
                        new_src_item.append(src_item[idx])
                        llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
                        llm_pos_ids_list.append(llm_pos_ids.to(position_ids.device))
                    elif src_item[idx] == image_token_id:
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                        llm_pos_ids = self._get_llm_pos_ids_for_vision(
                            start_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws)
                        llm_pos_ids_list.append(llm_pos_ids.to(position_ids.device))
                        vision_seqlen = image_grid_thw[image_idx].prod() // (spatial_merge_size ** 2)
                        new_src_item.extend([image_token_id] * vision_seqlen)
                        image_idx += 1
                    else:
                        T = video_grid_thw[video_idx][0].item()
                        H = video_grid_thw[video_idx][1].item()
                        W = video_grid_thw[video_idx][2].item()
                        llm_H = H // spatial_merge_size
                        llm_W = W // spatial_merge_size
                        tokens_per_frame = llm_H * llm_W
                        t_index_all = (torch.arange(T)).long()
                        start_pos = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                        current_pos = start_pos
                        final_frame_time = T - 1
                        for t in range(T):
                            if t != 0:
                                new_src_item.append(vision_start_token_id)
                                bot_pos = torch.full((3, 1), current_pos, dtype=torch.long)
                                llm_pos_ids_list.append(bot_pos.to(position_ids.device))
                                current_pos += 1
                            grid_h = torch.arange(llm_H).view(-1, 1).expand(-1, llm_W).flatten()
                            grid_w = torch.arange(llm_W).view(1, -1).expand(llm_H, -1).flatten()
                            frame_pos = torch.stack([
                                torch.full_like(grid_h, 0, dtype=torch.long),
                                grid_h, grid_w
                            ])
                            frame_pos_with_offset = frame_pos + current_pos
                            new_src_item.extend([video_token_id] * tokens_per_frame)
                            llm_pos_ids_list.append(frame_pos_with_offset.to(position_ids.device))
                            current_pos += max(llm_H, llm_W)
                            if t != final_frame_time:
                                new_src_item.append(vision_end_token_id)
                                eot_pos = torch.full((3, 1), current_pos, dtype=torch.long)
                                llm_pos_ids_list.append(eot_pos.to(position_ids.device))
                                current_pos += 1
                        video_idx += 1
                    idx += len(new_src_item) - new_src_item_len
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_delta = llm_positions.max() + 1 - len(total_input_ids[i])
                mrope_position_deltas.append(mrope_position_delta)
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def _get_llm_pos_ids_for_vision(self, start_idx, vision_idx, spatial_merge_size,
                                     t_index, grid_hs, grid_ws):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = (torch.arange(llm_grid_h).to(llm_grid_h.device)
                   .view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten())
        w_index = (torch.arange(llm_grid_w).to(llm_grid_h.device)
                   .view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten())
        t_index_tensor = (torch.Tensor(t_index).to(llm_grid_h.device)
                          .view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).long().flatten())
        _llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        video_embeds = self.visual.vision_projection(video_embeds)
        video_embeds = video_embeds.squeeze(0)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values, image_grid_thw=None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = self.visual.vision_projection(image_embeds)
        image_embeds = image_embeds.squeeze(0)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def forward(self, input_ids=None, attention_mask=None, mome_mask=None, swa_mask=None,
                position_ids=None, past_key_values=None, inputs_embeds=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None, rope_deltas=None,
                cache_position=None, second_per_grid_ts=None,
                direct_input=False, actual_seq_len=None,
                input_dump_dir=None, mask_dump_dir=None,
                **kwargs: Unpack[KwargsForCausalLM]):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0)
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}")
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if self.use_mhc:
            inputs_embeds = inputs_embeds.repeat(1, 1, self.mhc_num_stream)

        if input_ids is not None and (position_ids is None or position_ids.ndim == 2):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask)
            self.rope_deltas = rope_deltas

        if actual_seq_len is None:
            if attention_mask is not None:
                actual_seq_len = [attention_mask.shape[1]]
            else:
                assert position_ids.shape[1] == 1
                total_seq_len = position_ids.shape[-1]
                flat_pos_id = position_ids[0, 0]
                zero_idx = torch.where(flat_pos_id == 0)[0]
                actual_seq_len = zero_idx[1:].tolist()
                actual_seq_len.append(total_seq_len)

        attention_mask, swa_mask, mome_mask = calculate_masks(
            input_ids, actual_seq_len=actual_seq_len, mask_compress=True,
            apply_mome=self.config.text_config.use_mome,
            swa_layers=self.config.text_config.use_sliding_window,
            param_sink_number=self.config.text_config.param_sink_number)

        outputs = self.language_model(
            input_ids=None, position_ids=position_ids,
            attention_mask=attention_mask, swa_mask=swa_mask, mome_mask=mome_mask,
            actual_seq_len=actual_seq_len, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True, cache_position=cache_position,
            direct_input=kwargs.pop('direct_input', False), **kwargs,
        )

        output = VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            router_logits=outputs.router_logits,
        )
        return output if return_dict else output.to_tuple()


@dataclass
class VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None


class VL(PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual|lm_head))": "model.language_model",
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def _compute_mtp_loss(self, logits, labels, mtp_num_total_tokens, **kwargs):
        ignore_index = -100
        vocab_size = self.config.text_config.vocab_size
        mtp_loss_weight = self.config.text_config.mtp_loss_weight
        logits_per_head = logits.chunk(mtp_num_total_tokens, dim=0)
        head_losses = []
        for head_idx, head_logits in enumerate(logits_per_head):
            if head_idx == 0:
                head_labels = labels
            else:
                head_labels = torch.nn.functional.pad(
                    labels, (0, head_idx), value=ignore_index
                )[..., head_idx:]
            head_losses.append(
                self.loss_function(logits=head_logits, labels=head_labels, vocab_size=vocab_size, **kwargs))
        loss = head_losses[0]
        if mtp_num_total_tokens > 1:
            loss = loss + torch.stack(head_losses[1:]).mean() * mtp_loss_weight
        return loss

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None,
                rope_deltas=None, cache_position=None,
                second_per_grid_ts=None, **kwargs: Unpack[KwargsForCausalLM]):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs = self.model(
            input_ids=input_ids, pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=True,
            cache_position=cache_position, **kwargs,
        )

        hidden_states = outputs[0]
        mtp_num_total_tokens = self.config.text_config.mtp_num_total_tokens
        if mtp_num_total_tokens > 1:
            hidden_states = torch.cat(hidden_states.chunk(mtp_num_total_tokens, dim=-1), dim=0)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if mtp_num_total_tokens > 1:
                loss = self._compute_mtp_loss(logits, labels, mtp_num_total_tokens, **kwargs)
            else:
                loss = self.loss_function(logits=logits, labels=labels,
                                         vocab_size=self.config.text_config.vocab_size)
        return VLCausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, inputs_embeds=None,
                                      cache_position=None, position_ids=None,
                                      use_cache=True, pixel_values=None,
                                      pixel_values_videos=None,
                                      image_grid_thw=None, video_grid_thw=None,
                                      second_per_grid_ts=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position,
            position_ids=position_ids, pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts, use_cache=use_cache, **kwargs)
        model_inputs["position_ids"] = None
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs

    def _get_image_nums_and_video_nums(self, input_ids):
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)
        return image_nums, video_nums

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False,
                                      input_ids=None, **model_kwargs):
        if expand_size == 1:
            return input_ids, model_kwargs
        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos",
                       "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "image_grid_thw":
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(f"Expected value for key '{key}' to be a list")
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if key != "cache_position":
                    if (dict_to_expand[key] is not None
                            and isinstance(dict_to_expand[key], torch.Tensor)
                            and key not in visual_keys):
                        dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        model_kwargs = _expand_dict_for_generation(model_kwargs)
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])
        return input_ids, model_kwargs


__all__ = [
    "VLVisionConfig",
    "VLTextConfig",
    "VLConfig",
    "PreTrainedModel",
    "Model",
    "ForCausalLM",
    "PreTrainedModel",
    "VLModel",
    "VL",
]