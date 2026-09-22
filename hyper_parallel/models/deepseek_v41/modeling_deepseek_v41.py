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
from types import MethodType
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4PreTrainedModel,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)

from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.components.functional.sinkhorn import sinkhorn_knopps
from hyper_parallel.components.modules.engram import EngramModule, NgramHashMapping
from hyper_parallel.components.modules.mhc import PipelinedMhcModule, pipelined_mhc_post
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionCPContext as SharedAttentionCPContext,
    SharedCompressedPackedSequence as SharedPackedSequence,
    SharedCompressedAttentionState as SharedAttentionState,
    SharedCompressedDSAAttention,
    SharedCompressedDSAAttentionBase,
    SharedCompressedDSAIndexer,
    build_sliding_window_indices as _window_indices,
)
from hyper_parallel.models.deepseek_v41.adapter.data.image_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTower,
)

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
    """Initialize an embedding and clear its global padding row shard-safely."""
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.padding_idx is None:
        return
    weight = module.weight
    layout = getattr(weight, "layout", None)
    to_local = getattr(weight, "to_local", None)
    if layout is None or not callable(to_local):
        weight[module.padding_idx].zero_()
        return
    inner_rank = layout.rank_list.index(layout.mesh.rank)
    slice_area = infer_slice_area_by_layout(layout, inner_rank, weight.shape)
    row_start, row_end = slice_area[0]
    if row_start <= module.padding_idx < row_end:
        to_local()[module.padding_idx - row_start].zero_()


class DeepseekV41Engram(nn.Module):
    """Reference V4.1 Engram owned and executed by the model architecture."""

    def __init__(self, config: Any, layer_id: int, assets: dict[str, Any]) -> None:
        """Create the scaled table and the V4.1 gated residual projection."""
        super().__init__()
        self.layer_id = layer_id
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

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            segment_starts: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the released hash lookup and gated residual update."""
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


class DeepseekV41TopKRouter(nn.Module):
    """V4.1 learned router with independent text and image correction biases."""

    def __init__(self, config: Any) -> None:
        """Create the released noaux_tc routing parameter layout."""
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_local_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.scoring_func = str(config.scoring_func)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        if bool(getattr(config, "v41_vision_enabled", False)):
            self.bias_vl = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        else:
            self.register_parameter("bias_vl", None)

    def forward(
            self,
            hidden_states: torch.Tensor,
            image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw logits, routing weights, and selected experts.

        ``bias`` and ``bias_vl`` select experts only; the gathered routing
        weights intentionally use the unbiased scores, matching V4.1.
        """
        flattened = hidden_states.reshape(-1, self.hidden_size)
        logits = functional.linear(  # pylint: disable=not-callable
            flattened.float(), self.weight.float()
        )
        if self.scoring_func == "sqrtsoftplus":
            scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
        elif self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise ValueError(f"Unsupported V4.1 router scoring function: {self.scoring_func!r}")
        correction_bias = self.bias
        if image_mask is not None:
            if image_mask.shape != hidden_states.shape[:2]:
                raise ValueError("image_mask must have shape [batch, sequence]")
            if self.bias_vl is not None:
                correction_bias = torch.where(
                    image_mask.reshape(-1, 1),
                    self.bias_vl.unsqueeze(0),
                    self.bias.unsqueeze(0),
                )
        indices = torch.topk(scores + correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
        return logits, weights * self.routed_scaling_factor, indices


def _v41_sparse_moe_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the HF expert container with V4.1's visual router correction."""
    del input_ids
    batch_size, sequence_length, hidden_size = hidden_states.shape
    _, weights, indices = self.gate(hidden_states, image_mask=image_mask)
    routed = self.experts(hidden_states.view(-1, hidden_size), indices, weights)
    return routed.view(batch_size, sequence_length, hidden_size) + self.shared_experts(hidden_states)


class DeepseekV41Compressor(nn.Module):
    """Non-overlapping V4.1 compressed-KV producer."""

    def __init__(self, config: Any, compress_ratio: int) -> None:
        """Create the learned KV pooling projections."""
        super().__init__()
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        self.compress_ratio = compress_ratio
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        if compress_ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unrotated and RoPE-rotated compressed KV tensors."""
        batch_size, sequence_length, _ = hidden_states.shape
        if self.compress_ratio == 1:
            latent = self.norm(self.wkv(hidden_states))
            cos, sin = compress_position_embeddings
            rotated = apply_rotary_pos_emb(latent.unsqueeze(1), cos, sin).squeeze(1)
            return latent, rotated

        usable = sequence_length - sequence_length % self.compress_ratio
        key_value = self.wkv(hidden_states[:, :usable])
        gate = self.wgate(hidden_states[:, :usable])
        key_value = key_value.view(batch_size, -1, self.compress_ratio, key_value.shape[-1])
        gate = gate.view(batch_size, -1, self.compress_ratio, gate.shape[-1])
        latent = self.norm((key_value * gate.float().softmax(dim=2).to(key_value.dtype)).sum(dim=2))
        cos, sin = compress_position_embeddings
        cos = cos[:, :usable:self.compress_ratio]
        sin = sin[:, :usable:self.compress_ratio]
        rotated = apply_rotary_pos_emb(latent.unsqueeze(1), cos, sin).squeeze(1)
        return latent, rotated


class _DeepseekV41IndexerState(nn.Module):
    """Construct the released Full or Reindex Lightning Indexer state."""

    def __init__(self, config: Any, compress_ratio: int, layer_idx: int) -> None:
        """Create layer-local queries and Full-only shared-key projections."""
        super().__init__()
        self.compress_ratio = compress_ratio
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.owns_key = layer_idx in config.v41_kv_source_layer_ids
        candidate_source = int(getattr(config, "v41_candidate_source_layer_id", -1))
        self.is_candidate_source = layer_idx == candidate_source
        self.uses_candidates = 0 <= candidate_source < layer_idx
        self.candidate_topk_blocks = int(getattr(config, "v41_candidate_topk_blocks", 0))
        self.candidate_block_size = int(getattr(config, "v41_candidate_block_size", 1))
        self.loss_coeff = float(getattr(config, "v41_indexer_loss_coeff", 0.0))
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if self.owns_key:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)


class DeepseekV41Indexer(SharedCompressedDSAIndexer):
    """Reference Full/Reindex module owned by the V4.1 architecture."""

    def __init__(self, config: Any, compress_ratio: int, layer_idx: int) -> None:
        """Create the released Indexer state and attach its reference semantics."""
        super().__init__(_DeepseekV41IndexerState(config, compress_ratio, layer_idx))


class _DeepseekV41AttentionState(DeepseekV4Attention):
    """Construct the released V4.1 attention parameter tree."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Create V4 parameters plus the source-only compressor and indexer."""
        super().__init__(config, layer_idx)
        # DeepSeek-V4 applies this unweighted RMSNorm after q_b_proj. V4.1
        # explicitly removes it: q_norm remains between wq_a and wq_b, while
        # the projected query heads go directly into RoPE.
        del self.q_b_norm
        self.compress_ratio = int(config.v41_compress_ratios[layer_idx])
        self.is_kv_source = layer_idx in config.v41_kv_source_layer_ids
        self.is_index_source = layer_idx in config.v41_index_source_layer_ids
        kv_sources = [source for source in config.v41_kv_source_layer_ids if source <= layer_idx]
        index_sources = [source for source in config.v41_index_source_layer_ids if source <= layer_idx]
        self.kv_source_layer_idx = max(kv_sources, default=None)
        self.index_source_layer_idx = max(index_sources, default=None)
        candidate_source = int(getattr(config, "v41_candidate_source_layer_id", -1))
        self.candidate_source_layer_idx = (
            candidate_source if 0 <= candidate_source <= layer_idx else None
        )
        if self.is_kv_source:
            self.compressor = DeepseekV41Compressor(config, self.compress_ratio)
        if self.is_index_source:
            self.indexer = DeepseekV41Indexer(config, self.compress_ratio, layer_idx)


class DeepseekV41Attention(SharedCompressedDSAAttentionBase):
    """Reference shared compressed attention owned by the model architecture."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Create the canonical parameter tree with non-fused reference execution."""
        super().__init__(
            _DeepseekV41AttentionState(config, layer_idx),
            replace_indexer=False,
            use_optimized_sparse_attention=False,
        )


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
        """Compute V4.1 pre, post, and residual mixing coefficients."""
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
    elif isinstance(module, DeepseekV41TopKRouter):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        module.bias.zero_()
        if module.bias_vl is not None:
            module.bias_vl.zero_()
    elif isinstance(module, (DeepseekV41Attention, SharedCompressedDSAAttention)):
        module.sinks.zero_()
    elif isinstance(module, (DeepseekV41PipelinedHyperConnection, PipelinedMhcModule)):
        nn.init.normal_(module.fn, mean=0.0, std=std)
        module.base.zero_()
        module.scale.fill_(1.0)


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
        **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one V4.1 block while preserving the decoder module call boundary."""
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
    if image_mask is None:
        ffn_output = self.mlp(ffn_input, input_ids=input_ids)
    else:
        ffn_output = self.mlp(ffn_input, input_ids=input_ids, image_mask=image_mask)
    hidden_states = _hc_post(ffn_output, residual, ffn_post, ffn_comb)
    return hidden_states, ffn_pre


class DeepseekV41Model(DeepseekV4PreTrainedModel):
    """Depth-configurable V4.1 backbone with optional native vision support."""

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
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        for layer_idx in range(config.num_hidden_layers):
            layer = self.layers[layer_idx]
            layer.attn_hc = DeepseekV41PipelinedHyperConnection(layer.attn_hc)
            layer.ffn_hc = DeepseekV41PipelinedHyperConnection(layer.ffn_hc)
            layer.self_attn = DeepseekV41Attention(config, layer_idx)
            layer.forward = MethodType(_v41_decoder_layer_forward, layer)
            layer.mlp.gate = DeepseekV41TopKRouter(config)
            if bool(getattr(config, "v41_vision_enabled", False)):
                layer.mlp.forward = MethodType(_v41_sparse_moe_forward, layer.mlp)
        for layer_id in assets["layer_ids"]:
            self.layers[layer_id].engram = DeepseekV41Engram(config, layer_id, assets)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.vision = None
        self.aligner = None
        if bool(getattr(config, "v41_vision_enabled", False)):
            self.vision = DeepseekV41VisionTower(config)
            self.aligner = DeepseekV41VisionAligner(config)
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
        self.gradient_checkpointing = False
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize backbone state after FSDP has established local shards."""
        if isinstance(module, nn.Embedding):
            _initialize_embedding_shard_safe(module, self.config.initializer_range)
            return
        super()._init_weights(module)
        _initialize_v41_owned_module(module, self.config.initializer_range)
        if isinstance(module, DeepseekV41Model) and module.vision is not None:
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
        if self.vision is None or self.aligner is None:
            raise ValueError("received image inputs but V4.1 vision is disabled")
        image_count = image_vit_grid_hw.shape[0]
        required_shapes = {
            "image_patch_offsets": image_patch_offsets.numel() == image_count + 1,
            "image_llm_grid_hw": image_llm_grid_hw.shape == image_vit_grid_hw.shape,
            "image_batch_indices": image_batch_indices.numel() == image_count,
            "image_token_starts": image_token_starts.numel() == image_count,
        }
        invalid = [name for name, valid in required_shapes.items() if not valid]
        if invalid:
            raise ValueError(f"inconsistent V4.1 image metadata: {', '.join(invalid)}")
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [total_patches, 3, patch, patch]")
        image_records = []
        for image_index in range(image_count):
            patch_start = int(image_patch_offsets[image_index])
            patch_end = int(image_patch_offsets[image_index + 1])
            vit_height, vit_width = (int(value) for value in image_vit_grid_hw[image_index])
            llm_height, llm_width = (int(value) for value in image_llm_grid_hw[image_index])
            batch_index = int(image_batch_indices[image_index])
            token_start = int(image_token_starts[image_index])
            vision_features = self.vision(pixel_values[patch_start:patch_end], vit_height, vit_width)
            image_features = self.aligner(vision_features, vit_height, vit_width)
            if image_features.shape[0] != llm_height * llm_width:
                raise ValueError(
                    f"aligner/image-token count mismatch for image {image_index}: "
                    f"{image_features.shape[0]} versus {llm_height * llm_width}"
                )
            image_records.append((batch_index, token_start, llm_height, llm_width, image_features))
        return image_records

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
            image_sequence_start: Global start of this CP sequence shard.
            **kwargs: Additional decoder-layer keyword arguments.

        Returns:
            Decoder hidden state and optional cache metadata.
        """
        # The conditions below enforce one ordered forward contract across text,
        # vision, Engram, shared-attention, mHC, and gradient-checkpointing paths.
        #lizard forgives(cyclomatic_complexity)
        if use_cache or past_key_values is not None:
            raise NotImplementedError("the V4.1 validation crop supports training without KV cache")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is None:
            raise ValueError("input_ids are required while Engram is enabled")
        image_inputs = (
            pixel_values,
            image_patch_offsets,
            image_vit_grid_hw,
            image_llm_grid_hw,
            image_batch_indices,
            image_token_starts,
        )
        if any(value is not None for value in image_inputs):
            if token_types is None:
                raise ValueError("token_types are required with V4.1 image inputs")
            if any(value is None for value in image_inputs):
                raise ValueError("all V4.1 image metadata fields are required with pixel_values")
            image_records = self._encode_image_features(
                pixel_values, image_patch_offsets, image_vit_grid_hw, image_llm_grid_hw,
                image_batch_indices, image_token_starts,
            )
        else:
            image_records = []
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if image_records:
            inputs_embeds = self._merge_image_embeddings(
                inputs_embeds, token_types, image_records, image_sequence_start,
            )
        if token_types is not None and token_types.shape != input_ids.shape:
            raise ValueError("token_types must have the same shape as input_ids")
        image_mask = None if token_types is None else token_types >= 0
        engram_token_mask = None if image_mask is None else ~image_mask
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        segment_start_mask = None
        packed_sequence = kwargs.get("packed_seq_params")
        if packed_sequence is not None:
            if not isinstance(packed_sequence, SharedPackedSequence):
                raise TypeError(
                    "packed_seq_params must be SharedCompressedPackedSequence, "
                    f"got {type(packed_sequence).__name__}"
                )
            segment_start_positions = packed_sequence.local_segment_starts(input_ids.device)
            local_positions = torch.arange(
                packed_sequence.local_query_start,
                packed_sequence.local_query_start + packed_sequence.local_query_length,
                device=input_ids.device,
            ).unsqueeze(0)
            segment_start_mask = (local_positions == segment_start_positions).expand_as(input_ids)
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = hidden_states.new_zeros(*hidden_states.shape[:2], self.config.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        shared_state = SharedAttentionState()
        for layer in self.layers:
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
        hidden_states = self.norm(_hc_pre(hidden_states, pre_mix))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)


class DeepseekV41ForCausalLM(DeepseekV4ForCausalLM):
    """Causal-LM facade shared by full and validation-crop configurations."""

    def __init__(self, config: Any) -> None:
        """Create the configured backbone and unchanged causal-LM facade."""
        DeepseekV4PreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        self.model = DeepseekV41Model(config)
        self.model_mode = self.model.model_mode
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize V4.1-only state in addition to the inherited V4 modules."""
        if isinstance(module, nn.Embedding):
            _initialize_embedding_shard_safe(module, self.config.initializer_range)
            return
        super()._init_weights(module)
        _initialize_v41_owned_module(module, self.config.initializer_range)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "DeepseekV41ForCausalLM":
        """Construct the model from its translated V4.1 configuration."""
        del kwargs
        return cls(config)


__all__ = [
    "DeepseekV41Attention",
    "DeepseekV41Engram",
    "DeepseekV41ForCausalLM",
    "DeepseekV41Indexer",
    "DeepseekV41Model",
    "DeepseekV41PipelinedHyperConnection",
    "DeepseekV41TopKRouter",
    "SharedAttentionCPContext",
    "SharedAttentionState",
]
