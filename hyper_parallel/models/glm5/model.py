# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""GLM5 causal language model."""
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.glm5.attention import (
    GLM5GQAAttention,
    GLM5MLAAttention,
    GLM5OfficialMLAAttention,
)
from hyper_parallel.models.glm5.dsa import GLM5DSAIndexer
from hyper_parallel.models.glm5.moe import GLM5MoE
from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.modules.rmsnorm import RMSNorm
from hyper_parallel.models.modules.rope import RotaryEmbedding


def _init_glm5_moe_experts(module: GLM5MoE) -> None:
    nn.init.kaiming_uniform_(module.experts.gate_up_proj, a=5 ** 0.5)
    nn.init.kaiming_uniform_(module.experts.down_proj, a=5 ** 0.5)


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return a finite mean when a sequence shard has no valid labels."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    loss_sum = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="sum",
    )
    valid_tokens = (flat_labels != -100).sum().clamp_min(1)
    return loss_sum / valid_tokens


def prepare_glm5_batch(batch: dict, model: Any) -> dict:
    """Pad and shard sequence inputs for GLM5 context parallel training."""
    cp_size = getattr(model, "_cp_size", 1)
    if cp_size <= 1:
        return batch

    cp_rank = getattr(model, "_cp_rank", 0)
    input_ids = batch["input_ids"]
    labels = batch.get("labels")
    seq_len = input_ids.shape[1]
    pad_len = (-seq_len) % cp_size

    if pad_len:
        input_ids = F.pad(input_ids, (0, pad_len), value=0)
        if labels is not None:
            labels = F.pad(labels, (0, pad_len), value=-100)

    padded_len = input_ids.shape[1]
    local_len = padded_len // cp_size
    start = cp_rank * local_len
    end = start + local_len

    prepared = dict(batch)
    prepared["input_ids"] = input_ids[:, start:end].contiguous()
    prepared["position_ids"] = torch.arange(
        start, end, dtype=torch.long, device=input_ids.device,
    )

    if labels is not None:
        shifted_labels = F.pad(labels[:, 1:], (0, 1), value=-100)
        prepared["labels"] = shifted_labels[:, start:end].contiguous()

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            if pad_len:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            prepared["attention_mask"] = attention_mask[:, start:end].contiguous()
        elif attention_mask.ndim == 4:
            if pad_len:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=float("-inf"))
                if attention_mask.shape[-2] != 1:
                    attention_mask = F.pad(
                        attention_mask, (0, 0, 0, pad_len), value=float("-inf")
                    )
            query_slice = slice(None) if attention_mask.shape[-2] == 1 else slice(start, end)
            prepared["attention_mask"] = attention_mask[
                :, :, query_slice, start:end
            ].contiguous()

    return prepared


@dataclass
class GLM5Config:
    """GLM5 model configuration."""

    vocab_size: int = 154856
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    q_lora_rank: Optional[int] = None
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_dense_layers: int = 3
    moe_intermediate_size: int = 1024
    kv_lora_rank: int = 576
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    index_topk: int = 2048
    index_head_dim: int = 64
    index_n_heads: int = 16
    dsa_topk: int = 2048
    dsa_indexer_dim: int = 64
    attention_type: str = "gqa"
    use_dsa: bool = False
    moe_router_type: str = "softmax"
    n_shared_experts: int = 0
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    layer_types: Optional[List[str]] = field(default=None)

    def __post_init__(self):
        self._validate_positive_fields()
        self._validate_attention_shape()
        self._validate_expert_shape()
        self._validate_layer_types()

    def _validate_positive_fields(self) -> None:
        """Validate scalar fields that must stay positive."""
        for field_name in (
                "vocab_size", "hidden_size", "intermediate_size",
                "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "moe_intermediate_size",
                "kv_lora_rank", "qk_rope_head_dim", "v_head_dim",
                "index_topk", "index_head_dim", "index_n_heads",
                "dsa_topk", "dsa_indexer_dim", "max_position_embeddings"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.num_dense_layers < 0:
            raise ValueError("num_dense_layers must be non-negative")
        if self.q_lora_rank is not None and self.q_lora_rank <= 0:
            raise ValueError("q_lora_rank must be positive when set")
        if self.qk_nope_head_dim < 0:
            raise ValueError("qk_nope_head_dim must be non-negative")

    def _validate_attention_shape(self) -> None:
        """Validate attention dimensions and attention type."""
        expected_hidden = self.num_attention_heads * self.head_dim
        if self.hidden_size != expected_hidden:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_attention_heads * head_dim ({expected_hidden})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.attention_type == "mla" and self.qk_rope_head_dim > self.head_dim:
            raise ValueError("qk_rope_head_dim must be <= head_dim")
        if self.attention_type not in ("gqa", "mla", "glm_moe_dsa_mla"):
            raise ValueError(f"Unsupported GLM5 attention_type: {self.attention_type}")
        if self.attention_type == "glm_moe_dsa_mla":
            if self.q_lora_rank is None:
                raise ValueError("q_lora_rank is required for glm_moe_dsa_mla")
            if self.qk_nope_head_dim + self.qk_rope_head_dim != self.head_dim:
                raise ValueError(
                    "head_dim must equal qk_nope_head_dim + qk_rope_head_dim"
                )
            if self.index_head_dim < self.qk_rope_head_dim:
                raise ValueError("index_head_dim must be >= qk_rope_head_dim")

    def _validate_expert_shape(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_tok <= 0:
            raise ValueError("num_experts_per_tok must be positive")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must be <= num_experts")

    def _validate_layer_types(self) -> None:
        """Validate dense and MoE layer layout."""
        if not 0 <= self.num_dense_layers <= self.num_hidden_layers:
            raise ValueError("num_dense_layers must be in [0, num_hidden_layers]")
        if self.layer_types is None:
            self.layer_types = [
                "dense" if layer_idx < self.num_dense_layers else "moe"
                for layer_idx in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length {len(self.layer_types)} != "
                f"num_hidden_layers {self.num_hidden_layers}"
            )
        invalid_layer_types = [
            layer_type for layer_type in self.layer_types
            if layer_type not in ("dense", "moe")
        ]
        if invalid_layer_types:
            raise ValueError(
                f"Unsupported GLM5 layer_types: {invalid_layer_types}"
            )


class GLM5Decoder(nn.Module):
    """One GLM5 decoder layer: RMSNorm -> Attention -> RMSNorm -> MLP/MoE."""

    def __init__(
        self,
        config: GLM5Config,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        if config.attention_type == "gqa":
            self.rotary_emb = RotaryEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
            )
            self.self_attn = GLM5GQAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                qkv_bias=config.attention_bias,
                out_bias=config.attention_bias,
                rope=self.rotary_emb,
                rms_norm_eps=config.rms_norm_eps,
                use_dsa=config.use_dsa,
            )
        elif config.attention_type == "mla":
            self.self_attn = GLM5MLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                use_dsa=config.use_dsa,
            )
            self.rotary_emb = self.self_attn.rotary_emb
        else:
            self.self_attn = GLM5OfficialMLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                index_topk=config.index_topk,
                index_head_dim=config.index_head_dim,
                index_n_heads=config.index_n_heads,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
            )
            self.rotary_emb = self.self_attn.rotary_emb
        self.dsa_indexer = (
            GLM5DSAIndexer(
                hidden_size=config.hidden_size,
                indexer_dim=config.dsa_indexer_dim,
                topk=config.dsa_topk,
            )
            if config.use_dsa
            else None
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        if self.layer_type == "dense":
            self.mlp = SwiGLUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
            )
        elif self.layer_type == "moe":
            self.mlp = GLM5MoE(
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                router_type=config.moe_router_type,
                n_shared_experts=config.n_shared_experts,
                routed_scaling_factor=config.routed_scaling_factor,
                n_group=config.n_group,
                topk_group=config.topk_group,
                norm_topk_prob=config.norm_topk_prob,
            )
            _init_glm5_moe_experts(self.mlp)
        else:
            raise ValueError(
                f"Unknown GLM5 layer_type '{self.layer_type}' at layer {layer_idx}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Run one decoder layer."""
        del kwargs
        attention_past = past_key_value
        indexer_past_key = None
        if self.dsa_indexer is not None and past_key_value is not None:
            attention_past, indexer_past_key = past_key_value
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        topk_indices = None
        indexer_key_cache = None
        if self.dsa_indexer is not None:
            topk_indices, indexer_key_cache = self.dsa_indexer(
                hidden_states,
                position_ids,
                past_key=indexer_past_key,
            )
        attn_output = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=attention_past,
            use_cache=use_cache,
            topk_indices=topk_indices,
        )
        present_key_value = None
        if use_cache and isinstance(attn_output, tuple):
            hidden_states, present_key_value = attn_output
        else:
            hidden_states = attn_output
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if use_cache:
            if self.dsa_indexer is not None:
                present_key_value = (present_key_value, indexer_key_cache)
            return hidden_states, present_key_value
        return hidden_states


class GLM5TextModel(nn.Module):
    """Inner GLM5 decoder stack."""

    def __init__(self, config: GLM5Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GLM5Decoder(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def rotary_emb(self):
        return self.layers[0].rotary_emb


class GLM5ForCausalLM(nn.Module):
    """GLM5 causal language model for Trainer integration."""

    _tp_plan = {
        "*.self_attn.q_proj": "colwise",
        "*.self_attn.k_proj": "colwise",
        "*.self_attn.v_proj": "colwise",
        "*.self_attn.o_proj": "rowwise",
        "*.mlp.gate_proj": "colwise",
        "*.mlp.up_proj": "colwise",
        "*.mlp.down_proj": "rowwise",
    }
    _cp_modules = ["*.self_attn.attention_core"]
    _ep_modules = ["*.mlp.experts"]

    def __init__(self, config: GLM5Config):
        super().__init__()
        self.config = config
        self._cp_size = 1
        self._cp_rank = 0
        self.model = GLM5TextModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.tie_weights()

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def norm(self):
        return self.model.norm

    @property
    def rotary_emb(self):
        return self.model.rotary_emb

    def tie_weights(self) -> None:
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def _past_length(self, past_key_values: list) -> int:
        """Return cached sequence length from the first layer cache."""
        if not past_key_values or past_key_values[0] is None:
            return 0
        first_past = past_key_values[0]
        if (
            isinstance(first_past, tuple)
            and first_past
            and isinstance(first_past[0], tuple)
        ):
            return first_past[0][0].shape[1]
        if isinstance(first_past, tuple):
            return first_past[0].shape[1]
        return first_past.shape[1]

    def _default_position_ids(
        self,
        input_ids: torch.Tensor,
        seq_len: int,
        past_len: int,
    ) -> torch.Tensor:
        return torch.arange(
            past_len,
            past_len + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )

    def _forward_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: list,
        use_cache: bool,
    ) -> tuple:
        """Run all decoder layers and collect cache outputs when requested."""
        next_past_key_values = [] if use_cache else None
        for layer, layer_past in zip(self.model.layers, past_key_values):
            layer_output = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, present = layer_output
                next_past_key_values.append(present)
            else:
                hidden_states = layer_output
        return hidden_states, next_past_key_values

    def _loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute shifted causal LM loss, accounting for CP-prepared labels."""
        if labels is None:
            return None
        if self._cp_size > 1:
            shift_logits = logits.contiguous().float()
            shift_labels = labels.contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
        return _masked_cross_entropy(shift_logits, shift_labels)

    def _build_output(
        self,
        loss: Optional[torch.Tensor],
        logits: torch.Tensor,
        next_past_key_values: Optional[list],
        hidden_states: torch.Tensor,
        return_hidden_states: bool,
    ) -> dict:
        """Pack model outputs in the Trainer-compatible dictionary format."""
        output: dict[str, Any] = {"loss": loss, "logits": logits}
        if next_past_key_values is not None:
            output["past_key_values"] = next_past_key_values
        if return_hidden_states:
            output["hidden_states"] = hidden_states
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Run GLM5 causal LM forward."""
        return_hidden_states = kwargs.pop("return_hidden_states", False)
        del kwargs
        _, seq_len = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.model.layers)
        if position_ids is None:
            position_ids = self._default_position_ids(
                input_ids, seq_len, self._past_length(past_key_values)
            )

        input_embeds = self.model.embed_tokens(input_ids)
        hidden_states, next_past_key_values = self._forward_layers(
            input_embeds,
            position_ids,
            attention_mask,
            past_key_values,
            use_cache,
        )
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return self._build_output(
            self._loss(logits, labels),
            logits,
            next_past_key_values,
            hidden_states,
            return_hidden_states,
        )


__all__ = [
    "GLM5Config",
    "GLM5Decoder",
    "GLM5ForCausalLM",
    "GLM5TextModel",
]
