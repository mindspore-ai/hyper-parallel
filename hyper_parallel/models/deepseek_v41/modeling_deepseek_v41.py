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
"""Training-capable DeepSeek-V4.1 validation crop.

The public V4.1 repository contains an inference-only implementation. This
module reuses the Transformers 5.13 DeepSeek-V4 building blocks for the
unchanged text path and implements the V4.1-only Engram, cross-layer shared
compressed attention, and pipelined mHC semantics.
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
from hyper_parallel.components.modules.engram import NgramHashMapping
from hyper_parallel.components.modules.mhc import pipelined_mhc_post
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionCPContext as SharedAttentionCPContext,
    SharedCompressedPackedSequence as SharedPackedSequence,
    SharedCompressedAttentionState as SharedAttentionState,
    SharedCompressedDSAAttention as DeepseekV41SharedCompressedAttention,
    build_sliding_window_indices as _window_indices,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTower,
)


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


class DeepseekV41EngramPlaceholder(nn.Module):
    """Parameter-owning Engram whose forward is supplied by replacement."""

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
        """Fail when the required Engram replacement was not applied."""
        del hidden_states, input_ids, segment_starts, token_mask
        raise RuntimeError("DeepSeek-V4.1 Engram requires its module replacement")


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
        self.compress_ratio = compress_ratio
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.norm = DeepseekV4RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unrotated and RoPE-rotated compressed KV tensors."""
        batch_size, sequence_length, _ = hidden_states.shape
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


class DeepseekV41Indexer(nn.Module):
    """Parameter owner for a V4.1 Full or Reindex Lightning Indexer."""

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

    def forward(
            self,
            hidden_states: torch.Tensor,
            query_residual: torch.Tensor,
            latent: torch.Tensor,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            *,
            cp_context: SharedAttentionCPContext | None = None,
            query_offset: int = 0,
    ) -> torch.Tensor:
        """Fail because the high-performance replacement owns Indexer execution."""
        del hidden_states, query_residual, latent, compress_position_embeddings
        del cp_context, query_offset
        raise RuntimeError("DeepSeek-V4.1 Indexer requires its module replacement")


class DeepseekV41AttentionPlaceholder(DeepseekV4Attention):
    """Parameter-owning attention whose forward must be replaced by the adapter."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Create V4 parameters plus the source-only compressor and indexer."""
        super().__init__(config, layer_idx)
        self.compress_ratio = int(config.v41_compress_ratios[layer_idx])
        self.is_kv_source = layer_idx in config.v41_kv_source_layer_ids
        self.is_index_source = layer_idx in config.v41_index_source_layer_ids
        if self.is_kv_source:
            self.compressor = DeepseekV41Compressor(config, self.compress_ratio)
        if self.is_index_source:
            self.indexer = DeepseekV41Indexer(config, self.compress_ratio, layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fail when the required V4.1 forward replacement was not applied."""
        del hidden_states, position_embeddings, position_ids, attention_mask, past_key_values, kwargs
        raise RuntimeError("DeepSeek-V4.1 shared attention requires its module replacement")


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


class DeepseekV41CroppedModel(DeepseekV4PreTrainedModel):
    """Four-layer V4.1 backbone with optional native vision input support."""

    def __init__(self, config: Any) -> None:
        """Create the four-layer backbone and attach active V4.1 modules."""
        super().__init__(config)
        if config.num_hidden_layers != 4:
            raise ValueError("DeepseekV41CroppedModel requires exactly 4 decoder layers")
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
            layer.self_attn = DeepseekV41AttentionPlaceholder(config, layer_idx)
            layer.forward = MethodType(_v41_decoder_layer_forward, layer)
            if bool(getattr(config, "v41_vision_enabled", False)):
                layer.mlp.gate = DeepseekV41TopKRouter(config)
                layer.mlp.forward = MethodType(_v41_sparse_moe_forward, layer.mlp)
        for layer_id in assets["layer_ids"]:
            self.layers[layer_id].engram = DeepseekV41EngramPlaceholder(config, layer_id, assets)
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
        if isinstance(module, DeepseekV41EngramPlaceholder):
            module.q_weight.fill_(1.0)
            module.k_weight.fill_(1.0)
        elif isinstance(module, DeepseekV41TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            module.bias.zero_()
            if module.bias_vl is not None:
                module.bias_vl.zero_()
        elif isinstance(module, DeepseekV41CroppedModel) and module.vision is not None:
            nn.init.normal_(module.image_start, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.image_end, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.image_newline, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, NgramHashMapping):
            module.reset_parameters()

    def _merge_image_embeddings(
            self,
            input_embeddings: torch.Tensor,
            token_types: torch.Tensor,
            pixel_values: torch.Tensor,
            image_patch_offsets: torch.Tensor,
            image_vit_grid_hw: torch.Tensor,
            image_llm_grid_hw: torch.Tensor,
            image_batch_indices: torch.Tensor,
            image_token_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Replace V4.1 image token spans with vision/aligner features.

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
        merged = input_embeddings.clone()
        for image_index in range(image_count):
            patch_start = int(image_patch_offsets[image_index])
            patch_end = int(image_patch_offsets[image_index + 1])
            vit_height, vit_width = (int(value) for value in image_vit_grid_hw[image_index])
            llm_height, llm_width = (int(value) for value in image_llm_grid_hw[image_index])
            batch_index = int(image_batch_indices[image_index])
            token_start = int(image_token_starts[image_index])
            if not 0 <= batch_index < merged.shape[0]:
                raise ValueError(f"image_batch_indices[{image_index}] is outside the batch")
            span_length = llm_height * (llm_width + 1) + 2
            token_end = token_start + span_length
            if token_start < 0 or token_end > merged.shape[1]:
                raise ValueError(f"image token span {image_index} is outside input_ids")
            span_types = token_types[batch_index, token_start:token_end]
            image_slots = span_types == 1
            expected_types = torch.tensor(
                [0] + ([1] * llm_width + [2]) * llm_height + [3],
                device=span_types.device,
                dtype=span_types.dtype,
            )
            if not torch.equal(span_types, expected_types):
                raise ValueError(f"image token_types do not match V4.1 grid layout for image {image_index}")
            vision_features = self.vision(pixel_values[patch_start:patch_end], vit_height, vit_width)
            image_features = self.aligner(vision_features, vit_height, vit_width)
            if image_features.shape[0] != int(image_slots.sum()):
                raise ValueError(
                    f"aligner/image-token count mismatch for image {image_index}: "
                    f"{image_features.shape[0]} versus {int(image_slots.sum())}"
                )
            span_embeddings = merged[batch_index, token_start:token_end]
            span_embeddings[span_types == 0] = self.image_start.to(span_embeddings.dtype)
            span_embeddings[span_types == 2] = self.image_newline.to(span_embeddings.dtype)
            span_embeddings[span_types == 3] = self.image_end.to(span_embeddings.dtype)
            span_embeddings[image_slots] = image_features.to(span_embeddings.dtype)
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
            **kwargs: Any,
    ) -> MoeModelOutputWithPast:
        """Execute image injection, Engram, shared attention, and pipelined mHC."""
        if use_cache or past_key_values is not None:
            raise NotImplementedError("the V4.1 validation crop supports training without KV cache")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is None:
            raise ValueError("input_ids are required while Engram is enabled")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
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
            inputs_embeds = self._merge_image_embeddings(
                inputs_embeds,
                token_types,
                pixel_values,
                image_patch_offsets,
                image_vit_grid_hw,
                image_llm_grid_hw,
                image_batch_indices,
                image_token_starts,
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


class DeepseekV41CroppedForCausalLM(DeepseekV4ForCausalLM):
    """Causal-LM facade for the four-layer V4.1 validation backbone."""

    def __init__(self, config: Any) -> None:
        """Create the cropped backbone and unchanged causal-LM facade."""
        DeepseekV4PreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        self.model = DeepseekV41CroppedModel(config)
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
        if isinstance(module, DeepseekV41EngramPlaceholder):
            module.q_weight.fill_(1.0)
            module.k_weight.fill_(1.0)
        elif isinstance(module, DeepseekV41TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            module.bias.zero_()
            if module.bias_vl is not None:
                module.bias_vl.zero_()
        elif isinstance(module, NgramHashMapping):
            module.reset_parameters()

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "DeepseekV41CroppedForCausalLM":
        """Construct the validation model from its translated HF configuration."""
        del kwargs
        return cls(config)


__all__ = [
    "DeepseekV41AttentionPlaceholder",
    "DeepseekV41CroppedForCausalLM",
    "DeepseekV41EngramPlaceholder",
    "DeepseekV41SharedCompressedAttention",
    "DeepseekV41TopKRouter",
    "SharedAttentionCPContext",
    "SharedAttentionState",
]
