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
"""Qwen3.5 (dense) — hybrid linear-attention + gated-attention, no MoE.

Mirrors the upstream ``Qwen3_5ForConditionalGeneration`` text architecture
(``model_type='qwen3_5'``, e.g. ``Qwen/Qwen3.5-0.8B-Base``):

- N-layer text decoder, ``layer_types=[linear, linear, linear, full] * (N/4)``
  - ``linear_attention`` layers use :class:`GatedDeltaNet`
  - ``full_attention`` layers use :class:`GroupQueryAttention` with
    ``attn_output_gate=True`` and ``partial_rotary_factor=0.25``
- Every layer's MLP is a plain :class:`SwiGLUMLP` (gate / up / down).
  **No MoE, no shared-expert, no routing** — this is what distinguishes
  the Qwen3.5-dense family (0.8B / 2B / 4B / 9B / 27B Base) from the
  35B-A3B / 122B-A10B / 397B-A17B MoE family.
- mRoPE with ``mrope_section=[11,11,10]`` (interleaved) on the rotary
  64-dim subset of each 256-dim attention head.
- Residual-style RMSNorm ``(1 + weight) * normed`` (zeros init) — shared
  with Qwen3.5-MoE via :class:`hyper_parallel.models.qwen3_5_moe.model.
  Qwen3_5RMSNorm`.

Text-only forward; any ``model.visual.*`` keys in a HF checkpoint are
dropped by :mod:`hyper_parallel.models.qwen3_5.checkpoint`.
"""
# pylint: disable=C0103  # HF transformers class-name convention (Qwen3_5*)
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.attention import GroupQueryAttention
from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.modules.linear_attention import GatedDeltaNet
from hyper_parallel.models.modules.rope import MultiModalRotaryEmbedding


class Qwen3_5RMSNorm(nn.Module):
    """Residual-style RMSNorm used by Qwen3.5: ``(1.0 + weight) * normed``.

    Weight is stored as ``(real_scale - 1.0)`` and initialised to zeros
    (so an untrained layer is identity-norm). fp32 internal compute,
    returns in input dtype.

    Distinct from :class:`hyper_parallel.models.modules.RMSNorm` (which
    uses ``weight * normed`` with weight init to ones — the variant
    powering Qwen3 / Qwen3-VL-MoE / Llama). Do NOT swap them silently:
    the stored weights are not compatible across the two conventions.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
        input_dtype = x.dtype
        x_fp = x.float()
        normed = x_fp * torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + self.eps)
        out = normed * (1.0 + self.weight.float())
        return out.to(input_dtype)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Qwen3_5Config:
    """Qwen3.5 dense text configuration.

    Field names mirror the upstream ``text_config`` schema so the loader can
    map directly. Defaults match ``Qwen/Qwen3.5-0.8B-Base``; override the
    size-dependent fields for larger siblings (2B / 4B / 9B / 27B).
    """

    # ── core ──
    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    # ── full-attention specifics (Qwen3.5 uses attn_output_gate + partial RoPE) ──
    attn_output_gate: bool = True
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    full_attention_interval: int = 4

    # ── linear-attention (Gated DeltaNet) specifics ──
    linear_num_value_heads: int = 16
    linear_num_key_heads: int = 16
    linear_value_head_dim: int = 128
    linear_key_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # ── layer-by-layer dispatch ──
    layer_types: Optional[List[str]] = None  # populated in __post_init__ if None

    # ── multimodal ids (not used in text-only training; kept for loader parity) ──
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    def __post_init__(self):
        if self.layer_types is None:
            interval = self.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length {len(self.layer_types)} != "
                f"num_hidden_layers {self.num_hidden_layers}"
            )

# ============================================================================
# One hybrid dense layer
# ============================================================================

class Qwen3_5Decoder(nn.Module):
    """One Qwen3.5 dense layer: pre-norm → (linear_attn | self_attn) → MLP.

    Submodule names mirror the upstream module so the loader is a straight
    rename of checkpoint keys::

        input_layernorm.weight
        post_attention_layernorm.weight
        linear_attn.*              (layer_type == "linear_attention")
        self_attn.*                (layer_type == "full_attention")
        mlp.gate_proj.weight       (plain dense SwiGLU — no experts)
        mlp.up_proj.weight
        mlp.down_proj.weight
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int,
        rope: MultiModalRotaryEmbedding,
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_v_heads=config.linear_num_value_heads,
                num_k_heads=config.linear_num_key_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = GroupQueryAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                qkv_bias=config.attention_bias,
                out_bias=config.attention_bias,
                rope=rope,
                qk_norm=True,  # Qwen3.5 always has q_norm / k_norm
                rms_norm_eps=config.rms_norm_eps,
                attn_output_gate=config.attn_output_gate,
                norm_cls=Qwen3_5RMSNorm,  # (1+w) residual-style RMSNorm
            )
        else:
            raise ValueError(
                f"Unknown layer_type '{self.layer_type}' at layer {layer_idx}"
            )

        # Plain dense MLP (no experts, no routing) — THE distinguishing
        # feature vs Qwen3.5-MoE.
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
        )
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            # GatedDeltaNet ignores position_ids by design.
            hidden_states = self.linear_attn(
                hidden_states, attention_mask=attention_mask,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, position_ids=position_ids,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

# ============================================================================
# Top-level dense model
# ============================================================================

class Qwen3_5TextModel(nn.Module):
    """Inner text decoder — owns ``embed_tokens``, ``layers``, ``norm``, ``rotary_emb``.

    Two-level layout (``model: Qwen3_5TextModel`` + ``lm_head``) matches
    the standard Qwen3.5 module organisation; FSDP wrap units fall on
    ``model.layers[i]`` and the root ``Qwen3_5ForCausalLM``.
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rope_dim = int(config.head_dim * config.partial_rotary_factor)
        self.rotary_emb = MultiModalRotaryEmbedding(
            dim=rope_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )
        self.layers = nn.ModuleList([
            Qwen3_5Decoder(config, i, self.rotary_emb)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 dense causal LM — text-only.

    Submodule layout mirrors ``Qwen3_5ForConditionalGeneration``'s text
    decoder:

        model.embed_tokens.weight
        model.layers.{i}.*            (Qwen3_5Decoder)
        model.norm.weight
        lm_head.weight                (tied to embed_tokens when
                                       ``tie_word_embeddings=True``)
    """

    # Hyper-native model contract (see docs/model_authoring_contract.md §2).
    # Declared on the class so ``apply_tp`` / ``apply_cp`` discover them
    # without any external ``attach_*`` helper.
    _tp_plan = {
        "*.self_attn.q_proj": "colwise",
        "*.self_attn.k_proj": "colwise",
        "*.self_attn.v_proj": "colwise",
        "*.self_attn.o_proj": "rowwise",
        "*.mlp.gate_proj": "colwise",
        "*.mlp.up_proj": "colwise",
        "*.mlp.down_proj": "rowwise",
    }
    _cp_modules = ["*.self_attn"]

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config

        rope_dim = int(config.head_dim * config.partial_rotary_factor)
        if sum(config.mrope_section) * 2 != rope_dim:
            raise ValueError(
                f"sum(mrope_section)*2 ({sum(config.mrope_section) * 2}) "
                f"must equal rope_dim ({rope_dim} = head_dim * "
                f"partial_rotary_factor)"
            )
        self.model = Qwen3_5TextModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
        # Re-tie after ``to_empty`` — fresh per-Parameter storage breaks the
        # ``__init__``-time tie.
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device, dtype=torch.long,
            ).view(1, -1).expand(bsz, -1)

        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Right-pad labels with -100 (instead of slicing logits) so the
            # autograd graph flows through the full ``logits`` tensor;
            # slicing dispatches a different NPU kernel that drifts in fp32.
            logits_fp = logits.float()
            shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            loss = F.cross_entropy(
                logits_fp.view(-1, logits_fp.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
        return {"loss": loss, "logits": logits}

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5Decoder",
    "Qwen3_5ForCausalLM",
]
