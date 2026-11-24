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
"""Training-capable DeepSeek-V4.1 model implementation.

The public V4.1 repository contains an inference-only implementation. This
module reuses the Transformers 5.13 DeepSeek-V4 building blocks for the
unchanged text path and implements the V4.1-only Engram, cross-layer shared
compressed attention, and pipelined mHC semantics. The production-named model
class accepts both full and explicitly cropped configurations; current validation
assets exercise only ``v41_model_mode="validation_crop"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4PreTrainedModel,
    DeepseekV4RotaryEmbedding,
)

from hyper_parallel.components.functional.sinkhorn import sinkhorn_knopps
from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput
from hyper_parallel.components.modules import CausalLMHead
from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import CompressedSparseAttention2

# pylint: disable=unused-import
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.compressed_sparse_attention2 import (
    CompressedSparseAttention2Module,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.compressed_sparse_attention2 import (
    SharedCompressedAttentionState as SharedAttentionState,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.compressed_sparse_attention2 import (
    _apply_v41_rope,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.compressor import CSA2Compressor
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.document_layout import (
    SharedCompressedPackedSequence as SharedPackedSequence,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.indexer import (
    CSA2Indexer,
)
from hyper_parallel.components.modules.common.rms_norm import RMSNorm, RMSNormModule
from hyper_parallel.components.modules.common.swiglu_mlp import swiglu_activation
from hyper_parallel.components.modules.engram import Engram, EngramModule, NgramHashMapping
from hyper_parallel.components.modules.engram.host_engram_table import HostEngramTable
from hyper_parallel.components.modules.megakernel_utils import prepare_module_runtimes
from hyper_parallel.components.modules.mhc import PipelinedMhc, PipelinedMhcModule, pipelined_mhc_post
from hyper_parallel.models.deepseek_v41.components.dspark import DeepseekV41DSpark, DSpark
from hyper_parallel.models.deepseek_v41.components.modality_moe import ModalityMoE
from hyper_parallel.models.deepseek_v41.components.modality_router import (
    DeepseekV41TopKRouter,
)
from hyper_parallel.models.deepseek_v41.components.packed_mega_experts import DeepseekV41MegaMoeExperts
from hyper_parallel.models.deepseek_v41.components.vision_transformer import VisionTransformer, encode_image_features
from hyper_parallel.models.deepseek_v41.dataset_preprocessing.image_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTower,
)
from hyper_parallel.models.materialization import register_rebuildable_buffer

# Native model replacements are optional; an installed package's own failures propagate.
try:
    from hyper_multicore import MegaGate as CORE_MEGA_GATE
except ModuleNotFoundError as error:
    if error.name != "hyper_multicore":
        raise
    CORE_MEGA_GATE = None

_FULL_MODEL_MODE = "full"
_VALIDATION_CROP_MODE = "validation_crop"


def _resolve_v41_model_mode(config: Any) -> str:
    """Resolve and validate the explicit full-versus-crop construction mode."""
    model_mode = getattr(config, "v41_model_mode", _FULL_MODEL_MODE)
    if model_mode not in {_FULL_MODEL_MODE, _VALIDATION_CROP_MODE}:
        raise ValueError(
            "v41_model_mode must be 'full' or 'validation_crop', "
            f"got {model_mode!r}"
        )
    return model_mode


def _initialize_embedding_shard_safe(module: nn.Embedding, std: float) -> None:
    """Initialize an embedding and clear its global padding row through DTensor."""
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.padding_idx is None:
        return
    module.weight[module.padding_idx].zero_()


class DeepseekV41Engram(nn.Module):
    """Reference V4.1 Engram owned and executed by the model architecture."""

    def __init__(self, config: Any, layer_id: int, assets: dict[str, Any]) -> None:
        """Create the scaled table and the V4.1 gated residual projection."""
        super().__init__()
        self.layer_id = layer_id
        backend = getattr(config, "engram_storage_backend", "device")
        if backend not in ("device", "host"):
            raise ValueError("engram_storage_backend must be 'device' or 'host'")
        if getattr(config, "engram_host_lookup_impl", None) is not None:
            raise ValueError("Engram Host lookup implementation selection is unsupported")
        self.engram_max_pending_entries = int(getattr(config, "engram_max_pending_entries", 1000000))
        self.engram_max_sparse_rows_per_step = int(getattr(config, "engram_max_sparse_rows_per_step", 1000000))
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1.0e-6
        self.hash_mapping = NgramHashMapping(assets, layer_id)
        self.logical_num_embeddings = self.hash_mapping.logical_num_embeddings
        layer_index = assets["layer_ids"].index(layer_id)
        padded_sizes = assets.get("padded_num_embeddings")
        if padded_sizes is None:
            pad_multiple = int(getattr(config, "v41_engram_table_pad_multiple", 1))
            if pad_multiple < 1:
                raise ValueError("Engram table padding multiple must be positive")
            logical_size = int(assets["num_embeddings"][layer_index])
            self.padded_num_embeddings = (
                (logical_size + pad_multiple - 1) // pad_multiple * pad_multiple
            )
        else:
            self.padded_num_embeddings = int(padded_sizes[layer_index])
        if self.padded_num_embeddings < self.logical_num_embeddings:
            raise ValueError("Engram padded table cannot be smaller than its hash address space")
        head_dim = int(assets["head_dim"])
        hash_columns = (int(assets["max_ngram_size"]) - 1) * int(assets["num_heads"])
        self.embed = nn.Embedding(self.padded_num_embeddings, head_dim)
        self.wkv = nn.Linear(hash_columns * head_dim, self.hidden_size * (self.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size))

    def checkpoint_logical_shapes(self) -> dict[str, tuple[int, ...]]:
        """Declare Engram embedding logical dimensions for padded HF loading."""
        return {"embed.weight": (int(self.logical_num_embeddings), self.embed.embedding_dim)}

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            segment_starts: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the released hash lookup and gated residual update.

        Args:
            hidden_states: Decoder hidden rows to update.
            input_ids: Token IDs for Engram hash lookup.
            segment_starts: Optional packed sequence boundaries.
            token_mask: Optional mask excluding image tokens.
        """
        hash_ids = self.hash_mapping(input_ids, segment_starts, token_mask)
        embeddings = self.embed(hash_ids)
        key_value = self.wkv(embeddings.flatten(start_dim=-2))
        key, value = key_value.split(
            [self.hc_mult * self.hidden_size, self.hidden_size],
            dim=-1,
        )
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        hidden_fp32 = hidden_states.float()
        weight = self.q_weight.float() * self.k_weight.float()
        reciprocal_std = torch.rsqrt(hidden_fp32.square().mean(-1) + self.eps)
        reciprocal_std = reciprocal_std * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden_fp32 * weight * key).sum(-1)
        dot = dot * reciprocal_std * self.hidden_size**-0.5
        root = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -root, root))
        fused = (
            hidden_fp32 + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        ).to(hidden_states.dtype)
        if token_mask is None:
            return fused
        return torch.where(
            token_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1),
            fused,
            hidden_states,
        )








def deepseek_v41_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Apply native FP32 clamp and SwiGLU to separate expert projections.

    Args:
        gate: Gate projection output, either dense or expert-major.
        up: Up projection output with the same shape as ``gate``.
        limit: ``swiglu_limit``; nonpositive values disable clamping.

    Returns:
        FP32 activation. The caller restores its projection input dtype before
        the down projection, keeping this math shared by routed and shared experts.
    """
    return swiglu_activation(gate, up, limit=limit, activation_dtype="float32")


def _v41_shared_expert_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Use native activation math without replacing the HF projection modules."""
    intermediate = deepseek_v41_swiglu(
        module.gate_proj(hidden_states), module.up_proj(hidden_states), module.limit,
    )
    return module.down_proj(intermediate.to(hidden_states.dtype))


def bind_shared_expert_forward(shared_expert: nn.Module) -> None:
    """Bind native shared-expert math while preserving HF state and module hooks.

    Args:
        shared_expert: HF shared MLP whose ``limit`` comes from ``swiglu_limit``.
    """
    shared_expert.forward = MethodType(_v41_shared_expert_forward, shared_expert)


def _v41_sparse_moe_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        router_token_mask: torch.Tensor | None = None,
        router_sequence_ids: torch.Tensor | None = None,
        router_num_sequences: int | None = None,
) -> torch.Tensor:
    """Run V4.1 text/image routing with optional auxiliary token masking."""
    del input_ids
    batch_size, sequence_length, hidden_size = hidden_states.shape
    _, weights, indices = self.gate(
        hidden_states, image_mask=image_mask, token_mask=router_token_mask,
        sequence_ids=router_sequence_ids, num_sequences=router_num_sequences,
    )
    routed = self.experts(hidden_states.view(-1, hidden_size), indices, weights)
    return routed.view(batch_size, sequence_length, hidden_size) + self.shared_experts(hidden_states)


class DeepseekV41Compressor(CSA2Compressor):
    """Construct the model compressor with the CSA2 rotary convention."""

    def __init__(self, config: Any, compress_ratio: int) -> None:
        """Retain the original model construction API."""
        super().__init__(config, compress_ratio, apply_rope=_apply_v41_rope)


class DeepseekV41Indexer(CSA2Indexer):
    """Construct the model's canonical reference Indexer."""

    def __init__(self, config: Any, compress_ratio: int, layer_idx: int) -> None:
        """Create the released Full/Reindex projection tree."""
        super().__init__(config=config, compress_ratio=compress_ratio, layer_idx=layer_idx, apply_rope=_apply_v41_rope)


class DeepseekV41Attention(CompressedSparseAttention2Module):
    """Construct canonical reference attention before family selection."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Create CSA2 parameters with the original initialization order."""
        super().__init__(config=config, layer_idx=layer_idx, use_fused_ops=False)


class DeepseekV41PipelinedHyperConnection(nn.Module):
    """Reference pipelined mHC coefficient module owned by V4.1."""

    def __init__(self, module: nn.Module) -> None:
        """Adopt the V4 parameter layout without enabling fused kernels."""
        super().__init__()
        self.input_norm = module.input_norm
        self.fn = module.fn
        self.base = module.base
        self.scale = module.scale
        self.hc_mult = int(module.hc_mult)
        self.hc_sinkhorn_iters = int(module.hc_sinkhorn_iters)
        self.hc_eps = float(module.hc_eps)
        self.train(module.training)

    def forward(
            self,
            hidden_streams: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute V4.1 pre, post, and residual mixing coefficients.

        Args:
            hidden_streams: Hyper-connection streams entering the mixer.
        """
        flattened = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = functional.linear(flattened, self.fn.float())  # pylint: disable=not-callable
        num_stream = self.hc_mult
        pre, post, residual = mix.split(
            [num_stream, num_stream, num_stream * num_stream],
            dim=-1,
        )
        pre_bias, post_bias, residual_bias = self.base.split(
            [num_stream, num_stream, num_stream * num_stream]
        )
        pre_scale, post_scale, residual_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre * pre_scale + pre_bias) + self.hc_eps
        post = 2 * torch.sigmoid(post * post_scale + post_bias)
        residual = residual.view(*residual.shape[:-1], num_stream, num_stream)
        residual = residual * residual_scale + residual_bias.view(num_stream, num_stream)
        residual = sinkhorn_knopps(residual, self.hc_sinkhorn_iters, self.hc_eps)
        return pre, post, residual


def _initialize_v41_owned_module(module: nn.Module, std: float) -> None:
    """Preserve special initialization after V4.1 changes the HF source type."""
    if isinstance(module, (DeepseekV41Engram, EngramModule)):
        module.q_weight.fill_(1.0)
        module.k_weight.fill_(1.0)
    elif (isinstance(module, DeepseekV41TopKRouter)
          or (CORE_MEGA_GATE is not None and isinstance(module, CORE_MEGA_GATE))):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        tokens_per_expert = getattr(module, "tokens_per_expert", None)
        if tokens_per_expert is not None:
            tokens_per_expert.zero_()
        module.bias.zero_()
        if module.bias_vl is not None:
            module.bias_vl.zero_()
    elif isinstance(module, RMSNormModule):
        module.reset_parameters()
    elif isinstance(module, CompressedSparseAttention2Module):
        module.sinks.zero_()
    elif isinstance(module, (DeepseekV41PipelinedHyperConnection, PipelinedMhcModule)):
        nn.init.normal_(module.fn, mean=0.0, std=std)
        module.base.zero_()
        module.scale.fill_(1.0)
    elif isinstance(module, DeepseekV41DSpark):
        module.init_weights(std)


def _hc_pre(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the parallel residual streams."""
    return (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(dim=2).to(hidden_states.dtype)


def _hc_post(
        sublayer_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result through the reusable high-performance path."""
    return pipelined_mhc_post(sublayer_output, residual, post, comb)


def _identity_residual_mix(reference: torch.Tensor) -> torch.Tensor:
    """Build a batch-broadcast identity mapping for shifted mHC state."""
    num_stream = reference.shape[-1]
    identity = torch.eye(num_stream, dtype=reference.dtype, device=reference.device)
    view_shape = (1,) * (reference.ndim - 1) + (num_stream, num_stream)
    return identity.view(view_shape).expand(*reference.shape[:-1], num_stream, num_stream)


def _uses_shifted_mhc(layer: nn.Module) -> bool:
    """Validate and report whether both layer boundaries use the shifted protocol."""
    attention_shifted = callable(getattr(layer.attn_hc, "advance", None))
    ffn_shifted = callable(getattr(layer.ffn_hc, "advance", None))
    if attention_shifted != ffn_shifted:
        raise RuntimeError("attention and FFN mHC replacements must use the same protocol")
    return attention_shifted


def _v41_shifted_decoder_layer_forward(
        layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        pre_mix: torch.Tensor,
        input_ids: torch.LongTensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        shared_attention_state: SharedAttentionState,
        segment_starts: torch.Tensor | None,
        engram_token_mask: torch.Tensor | None,
        image_mask: torch.Tensor | None,
        previous_output: torch.Tensor | None,
        previous_post_mix: torch.Tensor | None,
        previous_residual_mix: torch.Tensor | None,
        router_token_mask: torch.Tensor | None,
        router_sequence_ids: torch.Tensor | None,
        router_num_sequences: int | None,
        **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """Run the shifted-mHC decoder path after protocol selection."""
    if previous_output is None or previous_post_mix is None or previous_residual_mix is None:
        raise ValueError("shifted mHC decoder state is incomplete")
    if hasattr(layer, "engram"):
        hidden_states = _hc_post(
            previous_output,
            hidden_states,
            previous_post_mix,
            previous_residual_mix,
        )
        hidden_states = layer.engram(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask=engram_token_mask,
        )
        previous_output = torch.zeros_like(previous_output)
        previous_post_mix = torch.zeros_like(previous_post_mix)
        previous_residual_mix = _identity_residual_mix(pre_mix)

    residual, attention_pre, attention_post, attention_comb, attention_input = layer.attn_hc.advance(
        previous_output,
        hidden_states,
        pre_mix,
        previous_post_mix,
        previous_residual_mix,
        layer.input_layernorm.weight,
    )
    attention_output, _ = layer.self_attn(
        attention_input,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        shared_attention_state=shared_attention_state,
        **kwargs,
    )
    residual, ffn_pre, ffn_post, ffn_comb, ffn_input = layer.ffn_hc.advance(
        attention_output,
        residual,
        attention_pre,
        attention_post,
        attention_comb,
        layer.post_attention_layernorm.weight,
    )
    ffn_output = layer.mlp(
        ffn_input,
        input_ids=input_ids,
        image_mask=image_mask,
        router_token_mask=router_token_mask,
        router_sequence_ids=router_sequence_ids,
        router_num_sequences=router_num_sequences,
    )
    return ffn_output, residual, ffn_pre, ffn_post, ffn_comb


def _v41_decoder_layer_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        *,
        pre_mix: torch.Tensor,
        input_ids: torch.LongTensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        shared_attention_state: SharedAttentionState,
        segment_starts: torch.Tensor | None,
        engram_token_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        previous_output: torch.Tensor | None = None,
        previous_post_mix: torch.Tensor | None = None,
        previous_residual_mix: torch.Tensor | None = None,
        router_token_mask: torch.Tensor | None = None,
        router_sequence_ids: torch.Tensor | None = None,
        router_num_sequences: int | None = None,
        **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """Run one V4.1 block while preserving the decoder module call boundary."""
    if _uses_shifted_mhc(self):
        return _v41_shifted_decoder_layer_forward(
            self,
            hidden_states,
            pre_mix=pre_mix,
            input_ids=input_ids,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            shared_attention_state=shared_attention_state,
            segment_starts=segment_starts,
            engram_token_mask=engram_token_mask,
            image_mask=image_mask,
            previous_output=previous_output,
            previous_post_mix=previous_post_mix,
            previous_residual_mix=previous_residual_mix,
            router_token_mask=router_token_mask,
            router_sequence_ids=router_sequence_ids,
            router_num_sequences=router_num_sequences,
            **kwargs,
        )

    if hasattr(self, "engram"):
        hidden_states = self.engram(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask=engram_token_mask,
        )
    residual = hidden_states
    attention_pre, attention_post, attention_comb = self.attn_hc(hidden_states)
    attention_input = self.input_layernorm(_hc_pre(hidden_states, pre_mix))
    attention_output, _ = self.self_attn(
        attention_input,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        shared_attention_state=shared_attention_state,
        **kwargs,
    )
    hidden_states = _hc_post(attention_output, residual, attention_post, attention_comb)

    residual = hidden_states
    ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_states)
    ffn_input = self.post_attention_layernorm(_hc_pre(hidden_states, attention_pre))
    ffn_output = self.mlp(
        ffn_input, input_ids=input_ids, image_mask=image_mask, router_token_mask=router_token_mask,
        router_sequence_ids=router_sequence_ids, router_num_sequences=router_num_sequences,
    )
    hidden_states = _hc_post(ffn_output, residual, ffn_post, ffn_comb)
    return hidden_states, ffn_pre


class DeepseekV41DecoderLayer(DeepseekV4DecoderLayer):
    """Construct V4.1 computation from complete public and model-owned families."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Retain native parameter paths and the decoder's activation-checkpoint call boundary."""
        # Initialize registries without allocating the inherited HF architecture a second time.
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.layer_idx = layer_idx
        self.self_attn = CompressedSparseAttention2(config=config, layer_idx=layer_idx, use_fused_ops=False)
        self.mlp = ModalityMoE(config=config)
        norm_config = SimpleNamespace(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config=norm_config, cast_order="after_weight", fp32_weight=True)
        self.post_attention_layernorm = RMSNorm(config=norm_config, cast_order="after_weight", fp32_weight=True)
        mhc_config = SimpleNamespace(num_stream=config.hc_mult, hidden_size=config.hidden_size,
                                    sinkhorn_iters=config.hc_sinkhorn_iters, hc_eps=config.hc_eps,
                                    norm_eps=config.rms_norm_eps, use_gamma=False,
                                    init_scale=1.0, init_method_std=config.initializer_range)
        self.attn_hc = PipelinedMhc(config=mhc_config)
        self.ffn_hc = PipelinedMhc(config=mhc_config)

    forward = _v41_decoder_layer_forward


class DeepseekV41Model(DeepseekV4PreTrainedModel):
    """Depth-configurable V4.1 backbone with optional native vision support."""

    _no_split_modules = ["DeepseekV41DecoderLayer"]

    def __init__(self, config: Any) -> None:
        """Create the configured backbone and attach active V4.1 modules."""
        super().__init__(config)
        self.model_mode = _resolve_v41_model_mode(config)
        if config.num_hidden_layers < 1:
            raise ValueError("DeepseekV41Model requires at least one decoder layer")
        if len(config.v41_compress_ratios) != config.num_hidden_layers:
            raise ValueError(
                "v41_compress_ratios must contain one entry per decoder layer, "
                f"got {len(config.v41_compress_ratios)} for {config.num_hidden_layers} layers"
            )
        shared_layer_count = sum(
            ratio > 0 and layer_idx not in config.v41_kv_source_layer_ids
            for layer_idx, ratio in enumerate(config.v41_compress_ratios)
        )
        config.num_kv_shared_layers = max(
            int(getattr(config, "num_kv_shared_layers", 0)),
            shared_layer_count,
        )
        assets_path = Path(config.v41_engram_assets_path)
        with assets_path.open("r", encoding="utf-8") as assets_file:
            assets = json.load(assets_file)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            DeepseekV41DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        for layer_id in getattr(config, "v41_engram_layer_ids", assets["layer_ids"]):
            layer_index = assets["layer_ids"].index(layer_id)
            rows = int(assets["num_embeddings"][layer_index])
            multiple = int(getattr(config, "v41_engram_table_pad_multiple", 1))
            if multiple < 1:
                raise ValueError("Engram table padding multiple must be positive")
            padded_sizes = assets.get("padded_num_embeddings")
            padded = (rows + multiple - 1) // multiple * multiple if padded_sizes is None else padded_sizes[layer_index]
            engram_config = SimpleNamespace(
                layer_id=layer_id, hidden_size=config.hidden_size, hc_mult=config.hc_mult,
                eps=config.rms_norm_eps, clamp_value=1e-6, assets=assets,
                embedding_dim=int(assets["head_dim"]), padded_num_embeddings=padded,
                engram_max_pending_entries=getattr(config, "engram_max_pending_entries", 1000000),
                engram_max_sparse_rows_per_step=getattr(config, "engram_max_sparse_rows_per_step", 1000000),
            )
            self.layers[layer_id].engram = Engram(
                config=engram_config, engram_storage_backend=getattr(config, "engram_storage_backend", "device"),
            )
        self.norm = RMSNorm(config=SimpleNamespace(hidden_size=config.hidden_size, eps=config.rms_norm_eps),
                            cast_order="after_weight", fp32_weight=True)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.vision = None
        self.aligner = None
        if bool(getattr(config, "v41_vision_enabled", False)):
            self.vision = VisionTransformer(config=config, module_fqn="model.vision")
            self.aligner = DeepseekV41VisionAligner(config)
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
        self.gradient_checkpointing = False
        self._megamoe_runtime_prepared = False
        self.post_init()

    def _prepare_megamoe_runtime(self) -> None:
        """Share serial expert workspace once after EP binding and before execution."""
        if self._megamoe_runtime_prepared:
            return
        experts = [layer.mlp.experts for layer in self.layers
                   if isinstance(layer.mlp.experts, DeepseekV41MegaMoeExperts)]
        DeepseekV41MegaMoeExperts.prepare_runtime_group(experts)
        prepare_module_runtimes(self)
        self._megamoe_runtime_prepared = True

    @torch.no_grad()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize backbone state after FSDP has established local shards."""
        if isinstance(module, HostEngramTable):
            return
        if isinstance(module, nn.Embedding):
            _initialize_embedding_shard_safe(module, self.config.initializer_range)
            return
        super()._init_weights(module)
        _initialize_v41_owned_module(module, self.config.initializer_range)
        if isinstance(module, DeepseekV41Model) and getattr(module, "image_start", None) is not None:
            nn.init.normal_(module.image_start, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.image_end, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.image_newline, mean=0.0, std=self.config.initializer_range)

    def _encode_image_features(
            self,
            pixel_values: torch.Tensor,
            image_patch_offsets: torch.Tensor,
            image_vit_grid_hw: torch.Tensor,
            image_llm_grid_hw: torch.Tensor,
            image_batch_indices: torch.Tensor,
            image_token_starts: torch.Tensor,
    ) -> list[tuple[int, int, int, int, torch.Tensor]]:
        """Run complete ViT and aligner computation before LLM embedding.

        The dataset supplies image metadata in global image order. Vision
        executes per image because V4.1 applies bidirectional attention only
        inside a single image grid; no image may attend to another image.
        """
        return encode_image_features(self.vision, self.aligner, pixel_values, image_patch_offsets,
                                     image_vit_grid_hw, image_llm_grid_hw, image_batch_indices, image_token_starts)

    def _merge_image_embeddings(
            self,
            input_embeddings: torch.Tensor,
            token_types: torch.Tensor,
            image_records: list[tuple[int, int, int, int, torch.Tensor]],
            image_sequence_start: int,
    ) -> torch.Tensor:
        """Apply the official type-based image insertion to local CP token spans."""
        merged = input_embeddings.clone()
        local_end = image_sequence_start + merged.shape[1]
        for image_index, (batch_index, token_start, llm_height, llm_width, image_features) in enumerate(image_records):
            if not 0 <= batch_index < merged.shape[0]:
                raise ValueError(f"image_batch_indices[{image_index}] is outside the batch")

            span_length = llm_height * (llm_width + 1) + 2
            token_end = token_start + span_length
            if token_start < 0:
                raise ValueError(f"image token span {image_index} has a negative start")

            intersection_start = max(token_start, image_sequence_start)
            intersection_end = min(token_end, local_end)
            if intersection_start < intersection_end:
                local_start = intersection_start - image_sequence_start
                local_stop = intersection_end - image_sequence_start
                feature_start = intersection_start - token_start
                feature_stop = intersection_end - token_start
                expected_types = torch.tensor(
                    [IMAGE_START] + ([IMAGE] * llm_width + [IMAGE_NEW_LINE]) * llm_height + [IMAGE_END],
                    device=token_types.device,
                    dtype=token_types.dtype,
                )
                local_types = token_types[batch_index, local_start:local_stop]
                if not torch.equal(local_types, expected_types[feature_start:feature_stop]):
                    raise ValueError(f"image token_types do not match V4.1 grid layout for image {image_index}")

                local_span = merged[batch_index, local_start:local_stop]
                local_span[local_types == IMAGE_START] = self.image_start.to(merged.dtype)
                local_span[local_types == IMAGE_END] = self.image_end.to(merged.dtype)
                local_span[local_types == IMAGE_NEW_LINE] = self.image_newline.to(merged.dtype)
                image_offsets = torch.arange(feature_start, feature_stop, device=token_types.device)
                image_offsets = image_offsets[local_types == IMAGE]
                feature_indices = image_offsets - 1 - (image_offsets - 1) // (llm_width + 1)
                local_span[local_types == IMAGE] = image_features[feature_indices].to(merged.dtype)
            # Keep every vision parameter in the graph even when this CP rank
            # owns none of an image's token types.
            image_grad_anchor = (
                image_features.sum() + self.image_start.sum() + self.image_end.sum() + self.image_newline.sum()
            )
            merged = merged + image_grad_anchor.to(merged.dtype) * 0.0
        return merged

    def _collect_image_records(
            self,
            token_types: torch.LongTensor | None,
            pixel_values: torch.Tensor | None,
            image_patch_offsets: torch.LongTensor | None,
            image_vit_grid_hw: torch.LongTensor | None,
            image_llm_grid_hw: torch.LongTensor | None,
            image_batch_indices: torch.LongTensor | None,
            image_token_starts: torch.LongTensor | None,
    ) -> list[tuple[int, int, int, int, torch.Tensor]]:
        """Validate the V4.1 image metadata bundle and encode image features."""
        image_inputs = (
            pixel_values,
            image_patch_offsets,
            image_vit_grid_hw,
            image_llm_grid_hw,
            image_batch_indices,
            image_token_starts,
        )
        if not any(value is not None for value in image_inputs):
            return []
        if token_types is None:
            raise ValueError("token_types are required with V4.1 image inputs")
        if any(value is None for value in image_inputs):
            raise ValueError("all V4.1 image metadata fields are required with pixel_values")
        return self._encode_image_features(
            pixel_values, image_patch_offsets, image_vit_grid_hw, image_llm_grid_hw,
            image_batch_indices, image_token_starts,
        )

    @staticmethod

    def _build_segment_start_mask(
            packed_sequence: Any | None,
            input_ids: torch.LongTensor,
    ) -> torch.Tensor | None:
        """Derive the packed-sequence segment-start mask for the local shard."""
        if packed_sequence is None:
            return None
        if not isinstance(packed_sequence, SharedPackedSequence):
            raise TypeError(
                "packed_seq_params must be SharedCompressedPackedSequence, "
                f"got {type(packed_sequence).__name__}"
            )
        return packed_sequence.segment_start_mask(input_ids.device).expand_as(input_ids)

    def _prepare_image_embeddings(
            self, *, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor | None,
            token_types: torch.LongTensor | None, image_inputs: dict[str, Any],
            image_sequence_start: int, encoder_parallel_plan: Any | None,
            encoder_parallel_inputs: Any | None = None,
    ) -> torch.Tensor:
        """Encode images and merge only records owned by this decoder rank."""
        if encoder_parallel_plan is None:
            image_records = self._collect_image_records(
                token_types, image_inputs["pixel_values"], image_inputs["image_patch_offsets"],
                image_inputs["image_vit_grid_hw"], image_inputs["image_llm_grid_hw"],
                image_inputs["image_batch_indices"], image_inputs["image_token_starts"],
            )
            vision_grad_anchor = None
        else:
            routed_records, vision_grad_anchor = encoder_parallel_plan.encode_and_route(
                image_inputs if encoder_parallel_inputs is None else encoder_parallel_inputs,
            )
            image_records = [
                (item.batch_index, item.decoder_token_start, *item.metadata["llm_hw"], feature)
                for item, feature in routed_records
            ]
        if token_types is not None and token_types.shape != input_ids.shape:
            raise ValueError("token_types must have the same shape as input_ids")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        merge_token_types = token_types
        merge_start = image_sequence_start
        if encoder_parallel_plan is not None:
            encoder_parallel_plan.decoder_layout.validate_hidden(inputs_embeds)
            if image_records:
                merge_token_types = encoder_parallel_plan.decoder_layout.localize(token_types)
                merge_start = encoder_parallel_plan.decoder_layout.global_start
        elif token_types is not None and inputs_embeds.shape[1] != token_types.shape[1]:
            # TP sequence parallelism has already reduced the embedding rows,
            # while token_types still describes the complete CP-local input.
            cp_length = token_types.shape[1]
            local_length = inputs_embeds.shape[1]
            local_offset = image_sequence_start % cp_length
            if local_offset + local_length > cp_length:
                raise ValueError("image sequence interval exceeds CP-local token types")
            merge_token_types = token_types[:, local_offset:local_offset + local_length]
        if image_records:
            inputs_embeds = self._merge_image_embeddings(
                inputs_embeds, merge_token_types, image_records, merge_start,
            )
        if encoder_parallel_plan is not None and encoder_parallel_plan.active:
            anchor = (vision_grad_anchor + self.image_start.sum() + self.image_end.sum()
                      + self.image_newline.sum())
            inputs_embeds = inputs_embeds + anchor.to(inputs_embeds.dtype) * 0.0
        return inputs_embeds

    def _prepare_decoder_inputs(self, input_ids, inputs_embeds, position_ids, kwargs):
        """Prepare rotary and packed-sequence metadata for the local decoder layers."""
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        packed_sequence = kwargs.get("packed_seq_params")
        if packed_sequence is not None:
            if not isinstance(packed_sequence, SharedPackedSequence):
                raise TypeError("packed_seq_params must be SharedCompressedPackedSequence")
            packed_sequence = packed_sequence.prepare(input_ids.device, tuple(self.config.v41_compress_ratios))
            kwargs["packed_seq_params"] = packed_sequence

            sequence_ids, sequence_capacity = packed_sequence.router_metadata(input_ids.device)
            kwargs["router_sequence_ids"] = sequence_ids
            kwargs["router_num_sequences"] = sequence_capacity
        segment_start_mask = self._build_segment_start_mask(packed_sequence, input_ids)
        return position_ids, position_embeddings, segment_start_mask

    def _run_decoder_layers(
            self,
            hidden_states: torch.Tensor,
            pre_mix: torch.Tensor,
            *,
            input_ids: torch.LongTensor,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None,
            shared_state: SharedAttentionState,
            segment_start_mask: torch.Tensor | None,
            engram_token_mask: torch.Tensor | None,
            image_mask: torch.Tensor | None,
            dspark_layer_ids: tuple[int, ...],
            pending_mhc: tuple[torch.Tensor, ...] = (),
            finish_mhc: bool = True,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Run decoder boundaries and collect detached DSpark targets.

        Args:
            hidden_states: Expanded residual streams.
            pre_mix: Input-mixing coefficients from the preceding boundary.
            input_ids: Token IDs used by Engram and expert routing.
            position_embeddings: Main and compressed rotary embeddings.
            position_ids: Token positions.
            attention_mask: Optional attention mask.
            shared_state: Shared compressed-attention state for this forward.
            segment_start_mask: Packed-sample boundary mask.
            engram_token_mask: Mask selecting tokens with Engram contributions.
            image_mask: Mask selecting visual tokens.
            dspark_layer_ids: Layers whose outputs supervise the drafter.
            pending_mhc: Shifted FFN output and mixing tensors from the previous stage.
            finish_mhc: Apply the last deferred FFN residual on the output stage.
            **kwargs: Additional decoder-layer keyword arguments.

        Returns:
            Completed residual streams and final input-mixing coefficients.
        """
        shifted_mhc_layers = [_uses_shifted_mhc(layer) for layer in self.layers.children()]
        if any(shifted_mhc_layers) and not all(shifted_mhc_layers):
            raise RuntimeError("shifted mHC must replace every retained V4.1 boundary")
        shifted_mhc = all(shifted_mhc_layers)
        previous_output = post_mix = residual_mix = None
        if pending_mhc:
            previous_output, post_mix, residual_mix = pending_mhc
        elif shifted_mhc:
            previous_output = torch.zeros_like(hidden_states[:, :, 0])
            post_mix = torch.zeros_like(pre_mix)
            residual_mix = _identity_residual_mix(pre_mix)
        for layer_index, layer in enumerate(self.layers.children()):
            if shifted_mhc:
                previous_output, hidden_states, pre_mix, post_mix, residual_mix = layer(
                    hidden_states,
                    pre_mix=pre_mix,
                    previous_output=previous_output,
                    previous_post_mix=post_mix,
                    previous_residual_mix=residual_mix,
                    input_ids=input_ids,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    shared_attention_state=shared_state,
                    segment_starts=segment_start_mask,
                    engram_token_mask=engram_token_mask,
                    image_mask=image_mask,
                    **kwargs,
                )
            else:
                hidden_states, pre_mix = layer(
                    hidden_states,
                    pre_mix=pre_mix,
                    input_ids=input_ids,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    shared_attention_state=shared_state,
                    segment_starts=segment_start_mask,
                    engram_token_mask=engram_token_mask,
                    image_mask=image_mask,
                    **kwargs,
                )
            if self.dspark_target_hiddens is not None and layer_index in dspark_layer_ids:
                # DSpark targets include the pending FFN post, but must not
                # propagate drafter gradients into the backbone.
                with torch.no_grad():
                    target_hidden = hidden_states
                    if shifted_mhc:
                        target_hidden = _hc_post(previous_output, hidden_states, post_mix, residual_mix)
                    self.dspark_target_hiddens.append(_hc_pre(target_hidden, pre_mix).detach())
        if shifted_mhc:
            if not finish_mhc:
                return hidden_states, pre_mix, previous_output, post_mix, residual_mix
            hidden_states = _hc_post(previous_output, hidden_states, post_mix, residual_mix)
        return hidden_states, pre_mix

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Any | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            token_types: torch.LongTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            image_patch_offsets: torch.LongTensor | None = None,
            image_vit_grid_hw: torch.LongTensor | None = None,
            image_llm_grid_hw: torch.LongTensor | None = None,
            image_batch_indices: torch.LongTensor | None = None,
            image_token_starts: torch.LongTensor | None = None,
            image_sequence_start: int = 0,
            encoder_parallel_plan: Any | None = None,
            encoder_parallel_inputs: Any | None = None,
            **kwargs: Any,
    ) -> MoeModelOutputWithPast:
        """Execute image injection, Engram, shared attention, and pipelined mHC.

        Args:
            input_ids: Input token IDs.
            attention_mask: Compact sample-boundary metadata.
            position_ids: Token position IDs.
            past_key_values: Unsupported KV-cache state.
            inputs_embeds: Precomputed token embeddings.
            use_cache: Whether to request KV-cache output.
            token_types: Text and image token-type IDs.
            pixel_values: Flattened ViT patch tensor.
            image_patch_offsets: Image offsets in ``pixel_values``.
            image_vit_grid_hw: Per-image ViT grid dimensions.
            image_llm_grid_hw: Per-image LLM grid dimensions.
            image_batch_indices: Batch index for each image.
            image_token_starts: Global token start for each image span.
            image_sequence_start: Global start of this decoder hidden shard.
            encoder_parallel_plan: Derived visual work and feature route for this batch.
            encoder_parallel_inputs: Device patches already packed for this encoder rank.
            **kwargs: Additional decoder-layer keyword arguments.

        Returns:
            Decoder hidden state and optional cache metadata.
        """
        if use_cache or past_key_values is not None:
            raise NotImplementedError("the V4.1 validation crop supports training without KV cache")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is None:
            raise ValueError("input_ids are required while Engram is enabled")
        self._prepare_megamoe_runtime()
        inputs_embeds = self._prepare_image_embeddings(
            input_ids=input_ids, inputs_embeds=inputs_embeds, token_types=token_types,
            image_inputs={
                "pixel_values": pixel_values,
                "image_patch_offsets": image_patch_offsets,
                "image_vit_grid_hw": image_vit_grid_hw,
                "image_llm_grid_hw": image_llm_grid_hw,
                "image_batch_indices": image_batch_indices,
                "image_token_starts": image_token_starts,
            },
            image_sequence_start=image_sequence_start, encoder_parallel_plan=encoder_parallel_plan,
            encoder_parallel_inputs=encoder_parallel_inputs,
        )
        image_mask = None if token_types is None else token_types >= 0
        engram_token_mask = None if image_mask is None else ~image_mask
        position_ids, position_embeddings, segment_start_mask = self._prepare_decoder_inputs(
            input_ids, inputs_embeds, position_ids, kwargs,
        )
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = hidden_states.new_zeros(*hidden_states.shape[:2], self.config.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        shared_state = SharedAttentionState()
        dspark_layer_ids = tuple(getattr(self.config, "v41_dspark_target_layer_ids", ()) or ())
        self.dspark_target_hiddens = [] if dspark_layer_ids else None
        self.dspark_inputs_embeds = inputs_embeds.detach() if dspark_layer_ids else None
        hidden_states, pre_mix = self._run_decoder_layers(
            hidden_states,
            pre_mix,
            input_ids=input_ids,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            shared_state=shared_state,
            segment_start_mask=segment_start_mask,
            engram_token_mask=engram_token_mask,
            image_mask=image_mask,
            dspark_layer_ids=dspark_layer_ids,
            **kwargs,
        )
        hidden_states = self.norm(_hc_pre(hidden_states, pre_mix))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)



class DeepseekV41CroppedForCausalLM(DeepseekV4ForCausalLM):
    """Causal-LM facade for the four-layer V4.1 validation backbone."""


class DeepseekV41ForCausalLM(DeepseekV4ForCausalLM):
    """Causal-LM facade shared by full and validation-crop configurations."""

    _no_split_modules = ["DeepseekV41DecoderLayer"]
    supports_chunked_loss = True

    def __init__(self, config: Any) -> None:
        """Create the configured backbone and unchanged causal-LM facade."""
        DeepseekV4PreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        self.model = DeepseekV41Model(config)
        self.model_mode = self.model.model_mode
        self.vocab_size = config.vocab_size
        self.lm_head = CausalLMHead(config=config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.dspark = None
        self.dspark_loss_coeff = 0.0
        if int(getattr(config, "v41_dspark_depth", 0) or 0) > 0:
            self.dspark = DSpark(config=config)
            self.dspark_loss_coeff = float(getattr(config, "v41_dspark_loss_coeff", 1.0))
        self.post_init()

    @torch.no_grad()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize V4.1-only state in addition to the inherited V4 modules."""
        if isinstance(module, HostEngramTable):
            return
        if isinstance(module, nn.Embedding):
            _initialize_embedding_shard_safe(module, self.config.initializer_range)
            return
        super()._init_weights(module)
        _initialize_v41_owned_module(module, self.config.initializer_range)

    def _forward_chunked(
            self, *, input_ids: torch.Tensor | None, targets: torch.Tensor,
            loss_mask: torch.Tensor | None, chunk_size: int, ignore_index: int, **kwargs: Any,
    ) -> ChunkedCausalLMOutput:
        """Compose the native backbone, terminal CE family and optional draft objective."""
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("V4.1 chunked training does not support KV cache")
        retained = kwargs.pop("logits_to_keep", 0)
        if isinstance(retained, torch.Tensor) or retained != 0 or kwargs.pop("return_dict", True) is False:
            raise ValueError("V4.1 chunked training requires complete tokens and return_dict=True")
        kwargs["use_cache"] = False
        outputs = self.model(input_ids=input_ids, **kwargs)
        terminal = self.lm_head(outputs.last_hidden_state, targets=targets, loss_mask=loss_mask,
                                chunk_size=chunk_size, ignore_index=ignore_index)
        draft_loss = None
        if self.dspark is not None:
            if input_ids is None:
                raise ValueError("DSpark chunked training requires input_ids")
            draft_targets = getattr(self.model, "dspark_target_hiddens", None)
            if not draft_targets:
                raise RuntimeError("DSpark requires collected backbone target hiddens")
            draft_loss, metrics = self.dspark(
                draft_targets, input_ids, targets, self.model.dspark_inputs_embeds, self.lm_head,
            )
            self.pending_extra_metrics = metrics
        return ChunkedCausalLMOutput(
            loss_sum=terminal.loss_sum, valid_token_count=terminal.valid_token_count,
            aux_loss=draft_loss, aux_loss_coef=self.dspark_loss_coeff,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            labels: torch.LongTensor | None = None,
            chunk_loss_targets: torch.Tensor | None = None,
            chunk_loss_mask: torch.Tensor | None = None,
            chunk_loss_chunk_size: int = 1024,
            chunk_loss_ignore_index: int = -100,
            **kwargs: Any,
    ) -> Any:  # pylint: disable=arguments-differ
        """Run the causal LM and, when enabled, add the DSpark objective.

        Args:
            input_ids: Input token ids forwarded to the V4 causal LM.
            labels: Source labels for the ordinary causal-LM objective.
            chunk_loss_targets: Pre-shifted targets for native memory-bounded CE.
            chunk_loss_mask: Optional token mask for the chunked objective.
            chunk_loss_chunk_size: Maximum tokens per terminal projection chunk.
            chunk_loss_ignore_index: Ignored target value in chunked training.
            **kwargs: Remaining Transformers forward arguments.

        Returns:
            The causal-LM output with the DSpark objective folded into
            ``loss`` and drafter metrics attached when enabled.
        """
        if chunk_loss_targets is not None:
            if labels is not None:
                raise ValueError("Pass pre-shifted chunk_loss_targets without source labels")
            return self._forward_chunked(input_ids=input_ids, targets=chunk_loss_targets,
                                         loss_mask=chunk_loss_mask, chunk_size=chunk_loss_chunk_size,
                                         ignore_index=chunk_loss_ignore_index, **kwargs)
        outputs = super().forward(input_ids=input_ids, labels=labels, **kwargs)
        # The trainer may keep labels on its loss-only path, so the drafter
        # must not depend on them: it supervises from input_ids when absent.
        if self.dspark is not None and input_ids is not None:
            targets = getattr(self.model, "dspark_target_hiddens", None)
            if not targets:
                raise RuntimeError("DSpark is enabled but no backbone target hiddens were collected")
            dspark_loss, dspark_metrics = self.dspark(
                targets, input_ids, labels, self.model.dspark_inputs_embeds, self.lm_head
            )
            dspark_term = self.dspark_loss_coeff * dspark_loss
            # The trainer's default ModelOutputLoss backpropagates
            # model_output.loss, so the drafter objective must join that
            # field with its graph attached; a logits-side auxiliary hook
            # would be dead because the HF loss is already built from the
            # original logits. The drafter share stays observable through
            # the dspark_* metrics.
            if outputs.loss is not None:
                outputs.loss = outputs.loss + dspark_term
            outputs.extra_metrics = dspark_metrics
            # Some output-object hand-offs rebuild the ModelOutput and drop
            # non-field attributes; the module attribute is the stable path.
            self.pending_extra_metrics = dspark_metrics
        return outputs

    @classmethod

    def from_config(cls, config: Any, **kwargs: Any) -> "DeepseekV41ForCausalLM":
        """Construct the model from its translated V4.1 configuration.

        Args:
            config: Translated DeepSeek-V4.1 model configuration.
            **kwargs: Native HF construction options, including dtype and attention implementation.
        """
        kwargs.pop("trust_remote_code", None)
        return cls._from_config(config, **kwargs)


__all__ = [
    "DeepseekV41Attention",
    "DeepseekV41Engram",
    "DeepseekV41ForCausalLM",
    "DeepseekV41Indexer",
    "DeepseekV41Model",
    "DeepseekV41PipelinedHyperConnection",
    "DeepseekV41TopKRouter",
    "SharedAttentionState",
]
