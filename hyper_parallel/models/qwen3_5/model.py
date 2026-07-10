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

Implements the ``Qwen3_5ForConditionalGeneration`` text architecture
(``model_type='qwen3_5'``, e.g. ``Qwen/Qwen3.5-0.8B-Base``):

- N-layer text decoder, ``layer_types=[linear, linear, linear, full] * (N/4)``
  - ``linear_attention`` layers use :class:`GatedDeltaNet`
  - ``full_attention`` layers use :class:`Qwen3_5Attention` with
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
# pylint: disable=C0103  # Qwen class-name convention (Qwen3_5*)
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.attention import _expand_kv_heads
from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.modules.linear_attention import torch_chunk_gated_delta_rule
from hyper_parallel.models.modules.rmsnorm import RMSNormGated
from hyper_parallel.models.modules.rope import MultiModalRotaryEmbedding, apply_rotary_pos_emb


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
        """Apply residual-style RMS normalization: ``(1 + weight) * normed``."""
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
# Attention core
# ============================================================================

class Qwen3_5SdpaCore(nn.Module):
    """Causal scaled-dot-product attention in BHSD layout — separate
    ``nn.Module`` so external model-level wrappers can target the kernel call.

    Delegates to :func:`torch.nn.functional.scaled_dot_product_attention`
    with ``is_causal=True`` when no additive mask is supplied.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """Run causal SDPA on ``[B, H, S, D]`` Q/K/V; returns ``[B, H, S, D]``."""
        sdpa_kwargs = {"enable_gqa": True} if enable_gqa else {}
        if attention_mask is not None:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=False, scale=scale, **sdpa_kwargs,
            )
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, **sdpa_kwargs)


def _normalize_qwen3_5_position_ids(
    position_ids: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return mRoPE position ids in the same form Transformers feeds RoPE."""
    if position_ids is None:
        position_ids = torch.arange(
            seq_len, device=device, dtype=torch.long,
        ).view(1, -1).expand(batch_size, -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:]
    return position_ids


def _prepare_qwen3_5_attention_masks(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(linear_attn_mask, causal_attn_mask)`` for hybrid layers."""
    if attention_mask is None:
        return None, None
    if attention_mask.ndim == 4:
        return None, attention_mask
    if attention_mask.ndim != 2:
        return attention_mask, attention_mask
    if torch.all(attention_mask == 1):
        return None, None
    linear_attention_mask = attention_mask.to(device=device)
    causal = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
    padding = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
    mask = causal[None, None, :, :] & padding
    if dtype == torch.bool:
        return linear_attention_mask, mask
    return linear_attention_mask, mask


def _apply_mask_to_padding_states(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Zero padding states before Qwen3.5 linear attention, matching Transformers."""
    if attention_mask is None or attention_mask.ndim != 2:
        return hidden_states
    if attention_mask.shape[1] <= 1 or attention_mask.shape[0] <= 1:
        return hidden_states
    return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)


# ============================================================================
# Qwen3.5-native attention (mRoPE + attn_output_gate + qk_norm + sdpa_core)
# ============================================================================

class Qwen3_5Attention(nn.Module):
    """Qwen3.5 full-attention layer.

    Uses the common GQA structure: q/k/v projections, optional Q/K norm, RoPE,
    SDPA, optional output gate, and ``o_proj``.

    Submodule names follow the common ``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``o_proj`` checkpoint convention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope: MultiModalRotaryEmbedding,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        attn_output_gate: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attn_output_gate = attn_output_gate

        q_out_dim = self.num_heads * self.head_dim * (2 if attn_output_gate else 1)
        self.q_proj = nn.Linear(hidden_size, q_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=out_bias)

        if qk_norm:
            self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rotary_emb = rope
        self.sdpa_core = Qwen3_5SdpaCore()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        bsz, seq_len, _ = hidden_states.shape

        q_raw = self.q_proj(hidden_states)
        k_raw = self.k_proj(hidden_states)
        v_raw = self.v_proj(hidden_states)

        if self.attn_output_gate:
            q_out = q_raw.reshape(bsz, seq_len, -1, self.head_dim * 2)
            q, gate = torch.chunk(q_out, 2, dim=-1)
            gate = gate.reshape(bsz, seq_len, -1)
            q = self.q_norm(q).transpose(1, 2)
        else:
            q = q_raw.reshape(bsz, seq_len, -1, self.head_dim)
            q = self.q_norm(q).transpose(1, 2)
            gate = None
        k = self.k_norm(
            k_raw.reshape(bsz, seq_len, -1, self.head_dim)
        ).transpose(1, 2)
        v = v_raw.reshape(bsz, seq_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kv_groups = q.shape[1] // k.shape[1]
        expand_kv_for_cp = bool(getattr(self, "_hp_cp_expand_kv_before_core", False))
        enable_gqa = kv_groups > 1 and attention_mask is None and not expand_kv_for_cp
        if kv_groups > 1 and (expand_kv_for_cp or not enable_gqa):
            k = _expand_kv_heads(k, kv_groups)
            v = _expand_kv_heads(v, kv_groups)

        attn_output = self.sdpa_core(
            q, k, v,
            attention_mask=attention_mask,
            scale=self.head_dim ** -0.5,
            enable_gqa=enable_gqa,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)

        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)


# ============================================================================
# One hybrid dense layer
# ============================================================================

class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet with a combined Q/K/V projection."""

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int = 32,
        num_k_heads: int = 16,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size
        self.kv_groups = num_v_heads // num_k_heads

        # Depthwise causal 1-D conv across the (Q, K, V) channel stack.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        a_init = torch.empty(num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(a_init))

        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj_input = nn.Identity()
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)
        bsz = hidden_states.shape[0]

        mixed_qkv_local = self.in_proj_qkv(hidden_states)
        mixed_qkv_local = mixed_qkv_local.transpose(1, 2)

        z_raw = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        seq_len = hidden_states.shape[1]

        # ``conv1d`` is depthwise (groups=conv_dim), so the channel stack has no
        # cross-channel interaction.
        mixed_qkv_local = F.silu(self.conv1d(mixed_qkv_local)[:, :, :seq_len])
        mixed_qkv_local = mixed_qkv_local.transpose(1, 2)

        key_dim_local = mixed_qkv_local.shape[-1] * self.key_dim // self.conv_dim
        value_dim_local = mixed_qkv_local.shape[-1] - 2 * key_dim_local
        query_local, key_local, value_local = torch.split(
            mixed_qkv_local,
            [key_dim_local, key_dim_local, value_dim_local],
            dim=-1,
        )
        n_k = query_local.shape[-1] // self.head_k_dim
        n_v = value_local.shape[-1] // self.head_v_dim
        z = z_raw.reshape(bsz, seq_len, n_v, self.head_v_dim)
        query_local = query_local.reshape(bsz, seq_len, n_k, self.head_k_dim)
        key_local = key_local.reshape(bsz, seq_len, n_k, self.head_k_dim)
        value_local = value_local.reshape(bsz, seq_len, n_v, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.kv_groups > 1:
            query_local = query_local.repeat_interleave(self.kv_groups, dim=2)
            key_local = key_local.repeat_interleave(self.kv_groups, dim=2)

        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query_local, key_local, value_local, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(bsz, seq_len, value_dim_local)

        core_attn_out = self.out_proj_input(core_attn_out)
        return self.out_proj(core_attn_out)


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
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_v_heads=config.linear_num_value_heads,
                num_k_heads=config.linear_num_key_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
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
        linear_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            # GatedDeltaNet ignores position_ids by design.
            hidden_states = self.linear_attn(
                hidden_states,
                attention_mask=linear_attention_mask,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
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

    Submodule layout follows ``Qwen3_5ForConditionalGeneration``'s text
    decoder:

        model.embed_tokens.weight
        model.layers.{i}.*            (Qwen3_5Decoder)
        model.norm.weight
        lm_head.weight                (tied to embed_tokens when
                                       ``tie_word_embeddings=True``)
    """

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
        """Return the decoder layer list."""
        return self.model.layers

    @property
    def embed_tokens(self):
        """Return the input token embedding layer."""
        return self.model.embed_tokens

    @property
    def norm(self):
        """Return the final RMS normalization layer."""
        return self.model.norm

    @property
    def rotary_emb(self):
        """Return the multi-modal rotary embedding used by full-attention layers."""
        return self.model.rotary_emb

    def tie_weights(self) -> None:
        """Re-tie lm_head and embed_tokens weights after parameter reinitialization."""
        # Re-tie after ``to_empty`` — fresh per-Parameter storage breaks the
        # ``__init__``-time tie.
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_ids: Token id tensor ``[B, S]``.
            labels: Optional ``[B, S]`` raw next-token labels; when present,
                returns a mean cross-entropy loss in ``out["loss"]``.
            position_ids: Optional ``[B, S]`` position ids. When ``None``,
                positions start at zero for the default eager path.
            attention_mask: Optional attention mask forwarded to decoder layers.
        """
        # pylint: disable=W0613  # interface conformance
        bsz, seq_len = input_ids.shape
        position_ids = _normalize_qwen3_5_position_ids(
            position_ids, bsz, seq_len, input_ids.device,
        )
        linear_attention_mask, causal_attention_mask = _prepare_qwen3_5_attention_masks(
            attention_mask, seq_len, self.model.embed_tokens.weight.dtype, input_ids.device,
        )

        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=causal_attention_mask,
                linear_attention_mask=linear_attention_mask,
            )
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Right-pad labels with -100 (instead of slicing logits) so the
            # autograd graph flows through the full ``logits`` tensor;
            # slicing dispatches a different NPU kernel that drifts in fp32.
            logits_fp = logits.float()
            targets = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = F.cross_entropy(
                logits_fp.view(-1, logits_fp.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            if num_items_in_batch is not None:
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
        return {"loss": loss, "logits": logits}

class Qwen3_5StageModule(nn.Module):
    """One pipeline stage's forward chunk.

    Holds the slice of :class:`Qwen3_5ForCausalLM` submodules that this
    stage owns:

    * **First stage** owns ``embed_tokens`` (consumes ``input_ids``) plus a
      contiguous layer range; its forward returns hidden states for the
      next stage.
    * **Intermediate stages** own only a layer range; their forward
      consumes and returns hidden states.
    * **Last stage** owns the trailing layer range, ``norm`` and
      ``lm_head``; its forward returns the scalar ``sum``-CE loss when
      ``targets`` are supplied (so the pipeline schedule's
      ``sens = ones_like(output)`` backward seed matches single-card
      ``loss.backward()`` semantics) and the logits tensor otherwise.

    Construction is performed by
    :func:`hyper_parallel.models.qwen3_5.parallelize.pipelining_qwen3_5`,
    which slices the assembled :class:`Qwen3_5ForCausalLM` and hands the
    relevant submodule references to this class.
    """

    def __init__(
        self,
        layers,
        embed_tokens: Optional[nn.Embedding] = None,
        rotary_emb: Optional[MultiModalRotaryEmbedding] = None,
        norm: Optional[Qwen3_5RMSNorm] = None,
        lm_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        if rotary_emb is not None:
            # Registered as a named submodule on every stage so the
            # per-layer ``Qwen3_5Attention`` references keep working
            # without a second copy of the rotary table.
            self.rotary_emb = rotary_emb
        self.layers = nn.ModuleList(layers)
        if norm is not None:
            self.norm = norm
        if lm_head is not None:
            self.lm_head = lm_head

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run this stage's forward chunk."""
        if hasattr(self, "embed_tokens"):
            bsz, seq_len = x.shape
            x = self.embed_tokens(x)
        else:
            bsz, seq_len, _ = x.shape
        position_ids = _normalize_qwen3_5_position_ids(
            position_ids, bsz, seq_len, x.device,
        )
        linear_attention_mask, causal_attention_mask = _prepare_qwen3_5_attention_masks(
            attention_mask, seq_len, x.dtype, x.device,
        )
        for layer in self.layers:
            x = layer(
                x,
                position_ids=position_ids,
                attention_mask=causal_attention_mask,
                linear_attention_mask=linear_attention_mask,
            )
        if hasattr(self, "norm"):
            x = self.norm(x)
        if not hasattr(self, "lm_head"):
            return x
        logits = self.lm_head(x)
        if targets is None:
            return logits
        logits_fp = logits.float()
        return F.cross_entropy(
            logits_fp.view(-1, logits_fp.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction="sum",
        )


__all__ = [
    "Qwen3_5Attention",
    "Qwen3_5Config",
    "Qwen3_5Decoder",
    "Qwen3_5ForCausalLM",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5SdpaCore",
    "Qwen3_5StageModule",
]
