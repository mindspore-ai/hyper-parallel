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
"""Qwen3.5-MoE (35B-A3B) — hybrid linear-attention + gated-attention + MoE.

Implements the ``Qwen3_5MoeForConditionalGeneration`` text architecture
configuration:

- 40-layer text decoder, ``layer_types=[linear, linear, linear, full] * 10``
  - ``linear_attention`` layers use :class:`Qwen3_5GatedDeltaNet`
  - ``full_attention`` layers use :class:`Qwen3_5Attention` with
    ``attn_output_gate=True`` and ``partial_rotary_factor=0.25``
- Every layer has the same MoE: 256 routed experts, top-8, plus 1 shared
  expert with sigmoid output gate (:class:`Qwen3_5SharedExpertMoE`).
- mRoPE with ``mrope_section=[11,11,10]`` (interleaved) on the rotary
  64-dim subset of each 256-dim attention head.

This module covers **text-only** training (no vision tower); the
loader silently drops vision keys when present in the checkpoint.
"""
# pylint: disable=C0103  # Qwen class-name convention (Qwen3_5*)
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.modules.rope import MultiModalRotaryEmbedding
from hyper_parallel.models.qwen3_5.model import Qwen3_5Attention, Qwen3_5GatedDeltaNet


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
class Qwen3_5MoeConfig:
    """Qwen3.5-MoE text configuration.

    Field names mirror the upstream ``text_config`` schema so the loader
    can map directly. Only fields read by the model are present; extras
    can be added without breaking the loader.
    """

    # ── core ──
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    # ── full-attention specifics ──
    attn_output_gate: bool = True
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    full_attention_interval: int = 4

    # ── linear-attention (Gated DeltaNet) specifics ──
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_value_head_dim: int = 128
    linear_key_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # ── MoE specifics ──
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False
    mtp_loss_weight: float = 0.0

    # ── layer-by-layer dispatch ──
    layer_types: Optional[List[str]] = None  # populated in __post_init__ if None

    # ── multimodal ids (used by VL builder; harmless for text-only) ──
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    def __post_init__(self):
        if self.layer_types is None:
            # Default Qwen3.5 pattern: 3 linear, 1 full, repeating.
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
    return linear_attention_mask, causal[None, None, :, :] & padding


class _Qwen3_5GroupedMM(torch.autograd.Function):
    """Per-expert grouped matmul used by Qwen3.5-MoE packed experts."""

    @staticmethod
    def forward(
        ctx,
        sorted_input: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        ctx.save_for_backward(sorted_input, weight)
        ctx.offsets = offsets
        output = torch.zeros(
            sorted_input.size(0), weight.size(2), device=sorted_input.device, dtype=sorted_input.dtype,
        )
        start = 0
        for expert_idx, end in enumerate(offsets.tolist()):
            if start != end:
                torch.mm(sorted_input[start:end], weight[expert_idx], out=output[start:end])
            start = end
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass."""
        sorted_input, weight = ctx.saved_tensors
        grad_input = torch.zeros_like(sorted_input)
        grad_weight = torch.zeros(
            weight.shape, device=weight.device, dtype=weight.dtype,
        )
        start = 0
        for expert_idx, end in enumerate(ctx.offsets.tolist()):
            if start != end:
                torch.mm(grad_output[start:end], weight[expert_idx].T, out=grad_input[start:end])
                torch.mm(sorted_input[start:end].T, grad_output[start:end], out=grad_weight[expert_idx])
            start = end
        return grad_input, grad_weight, None


def _qwen3_5_grouped_mm(
    sorted_input: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3.5-MoE grouped matmul on already expert-major tokens."""
    return _Qwen3_5GroupedMM.apply(sorted_input, weight, offsets)


class Qwen3_5MoeTopKRouter(nn.Module):
    """Top-k router matching Qwen3.5-MoE softmax-then-topk routing."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """Route tokens via softmax-then-topk with renormalization."""
        flat = hidden_states.reshape(-1, self.hidden_size)
        router_logits = F.linear(flat, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_value, top_index = torch.topk(router_probs, self.top_k, dim=-1)
        top_value /= top_value.sum(dim=-1, keepdim=True)
        top_value = top_value.to(router_logits.dtype)
        return router_logits, top_value, top_index


class Qwen3_5MoeExperts(nn.Module):
    """Qwen3.5-MoE packed routed experts with checkpoint grouped-mm ordering."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize packed expert projections for from-scratch construction."""
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

    @staticmethod
    def _expert_offsets(
        expert_ids_sorted: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Return cumulative per-expert token offsets."""
        device = expert_ids_sorted.device
        histc_input = expert_ids_sorted.float() if device.type == "cpu" else expert_ids_sorted.int()
        tokens_per_expert = torch.histc(
            histc_input, bins=num_experts, min=0, max=num_experts - 1,
        )
        return torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass matching HF grouped-mm fallback token ordering."""
        num_tokens, hidden_dim = hidden_states.shape
        num_top_k = top_k_index.size(-1)
        device = hidden_states.device

        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1).expand(-1, num_top_k).reshape(-1)
        )
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)

        invalid_mask = expert_ids >= self.num_experts
        expert_ids = expert_ids.clamp(0, self.num_experts - 1)

        perm = torch.argsort(expert_ids)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)

        expert_ids_sorted = expert_ids[perm]
        sample_weights_sorted = sample_weights[perm]
        sorted_hidden = hidden_states[token_idx[perm]]
        offsets = self._expert_offsets(expert_ids_sorted, self.num_experts)

        gate_up = _qwen3_5_grouped_mm(
            sorted_hidden, self.gate_up_proj.transpose(-2, -1), offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        down = _qwen3_5_grouped_mm(
            intermediate, self.down_proj.transpose(-2, -1), offsets,
        )

        if getattr(self, "_hp_moe_tp_enabled", False):
            weighted = down.to(torch.float32) * sample_weights_sorted.to(torch.float32).unsqueeze(-1)
        else:
            weighted = down * sample_weights_sorted.unsqueeze(-1)
        weighted.masked_fill_(invalid_mask[perm].unsqueeze(-1), 0.0)

        unsorted = weighted[inv_perm]
        final_hidden_states = unsorted.view(
            num_tokens, num_top_k, hidden_dim,
        ).sum(dim=1)
        if getattr(self, "_hp_moe_tp_enabled", False):
            return final_hidden_states.to(hidden_states.dtype)
        return final_hidden_states.to(hidden_states.dtype)

    def grouped_forward(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        gate_up: Optional[torch.Tensor] = None,
        down: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run experts on already dispatched, expert-major tokens."""
        if gate_up is None:
            gate_up = self.gate_up_proj
        if down is None:
            down = self.down_proj
        offsets = torch.cumsum(num_tokens_per_expert, dim=0).to(torch.int32)
        gate_up_out = _qwen3_5_grouped_mm(routed_input, gate_up.transpose(-2, -1), offsets)
        gate_part, up_part = gate_up_out.chunk(2, dim=-1)
        intermediate = F.silu(gate_part) * up_part
        return _qwen3_5_grouped_mm(intermediate, down.transpose(-2, -1), offsets)


class Qwen3_5SharedExpertMoE(nn.Module):
    """Qwen3.5-MoE block with routed packed experts plus one shared expert."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        shared_expert_intermediate_size: int,
        shared_expert_cls: type = SwiGLUMLP,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Qwen3_5MoeTopKRouter(hidden_size, num_experts, top_k)
        self.experts = Qwen3_5MoeExperts(num_experts, hidden_size, moe_intermediate_size)
        self.shared_expert = shared_expert_cls(
            hidden_size, shared_expert_intermediate_size, bias=False,
        )
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.shared_expert_gate.weight)
        self.router_logits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bsz, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        shared_out = self.shared_expert(x_flat)
        router_logits, top_value, top_index = self.gate(x_flat)
        self.router_logits = router_logits
        routed = self.experts(x_flat, top_index, top_value)
        shared_out = torch.sigmoid(self.shared_expert_gate(x_flat)) * shared_out
        return (routed + shared_out).view(bsz, seq_len, hidden_size)


# ============================================================================
# One hybrid layer
# ============================================================================


class Qwen3_5MoeDecoder(nn.Module):
    """One Qwen3.5-MoE layer: pre-norm → (linear_attn | self_attn) → MoE.

    Submodule names mirror the upstream module so the loader is a straight
    rename of checkpoint keys::

        input_layernorm.weight
        post_attention_layernorm.weight
        linear_attn.*           (when layer_type == "linear_attention")
        self_attn.*             (when layer_type == "full_attention")
        mlp.gate.weight                     (num_experts, hidden_size)
        mlp.experts.{e}.gate_up_proj        (2*moe_intermediate_size, hidden_size)
        mlp.experts.{e}.down_proj.weight    (hidden_size, moe_intermediate_size)
        mlp.shared_expert.{gate,up,down}_proj.weight
        mlp.shared_expert_gate.weight       (1, hidden_size)
    """

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
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
                qk_norm=True,  # Qwen3.5 always has q_norm/k_norm
                rms_norm_eps=config.rms_norm_eps,
                attn_output_gate=config.attn_output_gate,
            )
        else:
            raise ValueError(
                f"Unknown layer_type '{self.layer_type}' at layer {layer_idx}"
            )

        self.mlp = Qwen3_5SharedExpertMoE(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
        )
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        compute_dtype = (
            self.linear_attn.in_proj_qkv.weight.dtype
            if self.layer_type == "linear_attention"
            else self.self_attn.q_proj.weight.dtype
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(compute_dtype)
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
        hidden_states = hidden_states.to(compute_dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5MoeTextModel(nn.Module):
    """Inner text decoder — owns ``embed_tokens``, ``layers``, ``norm``, ``rotary_emb``.

    Two-level layout (``model: Qwen3_5MoeTextModel`` + ``lm_head``)
    matches the standard Qwen3.5-MoE module organisation; FSDP wrap units
    fall on ``model.layers[i]`` and the root ``Qwen3_5MoeForCausalLM``.
    """

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rope_dim = int(config.head_dim * config.partial_rotary_factor)
        self.rotary_emb = MultiModalRotaryEmbedding(
            dim=rope_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )
        # Per-layer rotary instances (one non-persistent inv_freq buffer
        # each): a single shared module registered under every full-attention
        # layer makes the activation-checkpoint wrapper see overlapping wrap
        # regions once two such layers exist.
        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoder(config, i, MultiModalRotaryEmbedding(
                dim=rope_dim,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
                mrope_section=config.mrope_section,
            ))
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        return_prenorm: bool = False,
        **kwargs,
    ):
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        position_ids = _normalize_qwen3_5_position_ids(
            position_ids, bsz, seq_len, inputs_embeds.device,
        )
        linear_attention_mask, causal_attention_mask = _prepare_qwen3_5_attention_masks(
            attention_mask, seq_len, inputs_embeds.device,
        )

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=causal_attention_mask,
                linear_attention_mask=linear_attention_mask,
            )
        if return_prenorm:
            return self.norm(hidden_states), hidden_states
        return self.norm(hidden_states)


def load_balancing_loss_func(
    gate_logits,
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
):
    """Qwen3.5-MoE load-balancing auxiliary loss.

    Args:
        gate_logits: Tuple of per-layer raw router logits, each shaped
            ``(num_tokens, num_experts)``.
        num_experts: Total number of experts.
        top_k: Number of experts routed per token.
        attention_mask: Optional ``(batch, seq)`` mask used to drop padding.

    Returns:
        The auxiliary loss scalar (``0`` when ``gate_logits`` is not a tuple).
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0,
    )

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0,
        ) / torch.sum(expert_attention_mask, dim=0)

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0,
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def collect_router_logits(layers) -> tuple:
    """Gather per-layer raw router logits stashed by each MoE block.

    Walks ``layers`` in order and reads ``layer.mlp.router_logits`` (set by
    :class:`Qwen3_5SharedExpertMoE` each forward), transparently unwrapping an
    activation-checkpoint wrapper if present.

    Args:
        layers: Iterable of decoder layers.

    Returns:
        Tuple of router-logit tensors in layer order (empty if none present).
    """
    collected = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            inner = getattr(layer, "_checkpoint_wrapped_module", layer)
            mlp = getattr(inner, "mlp", None)
        router_logits = getattr(mlp, "router_logits", None) if mlp is not None else None
        if router_logits is not None:
            collected.append(router_logits)
    return tuple(collected)


def moe_aux_loss(layers, config, attention_mask: Optional[torch.Tensor]):
    """Compute ``router_aux_loss_coef * load_balancing_loss`` for a backbone.

    Args:
        layers: Decoder layers carrying stashed ``mlp.router_logits``.
        config: A :class:`Qwen3_5MoeConfig` (num_experts / num_experts_per_tok
            / router_aux_loss_coef).
        attention_mask: Optional ``(batch, seq)`` padding mask.

    Returns:
        The scaled auxiliary-loss tensor, or ``None`` when no router logits
        were collected.
    """
    gate_logits = collect_router_logits(layers)
    if not gate_logits:
        return None
    aux = load_balancing_loss_func(
        gate_logits,
        config.num_experts,
        config.num_experts_per_tok,
        attention_mask,
    )
    return config.router_aux_loss_coef * aux


class Qwen3_5MoeForCausalLM(nn.Module):
    """Qwen3.5-MoE causal LM — text-only entry point.

    Submodule layout follows ``Qwen3_5MoeForConditionalGeneration``'s
    text decoder:

        model.embed_tokens.weight
        model.layers.{i}.*           (Qwen3_5MoeDecoder)
        model.norm.weight
        lm_head.weight
    """

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.config = config

        rope_dim = int(config.head_dim * config.partial_rotary_factor)
        if sum(config.mrope_section) * 2 != rope_dim:
            raise ValueError(
                f"sum(mrope_section)*2 ({sum(config.mrope_section) * 2}) "
                f"must equal rope_dim ({rope_dim} = head_dim * "
                f"partial_rotary_factor)"
            )
        self.model = Qwen3_5MoeTextModel(config)
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
            attention_mask, seq_len, input_ids.device,
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
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            # Right-pad labels with -100 (instead of slicing logits) so the
            # autograd graph flows through the full ``logits`` tensor;
            # slicing dispatches a different NPU kernel.
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
            if self.config.output_router_logits:
                aux = moe_aux_loss(self.model.layers, self.config, linear_attention_mask)
                if aux is not None:
                    loss = loss + aux.to(loss.device)
        return {"loss": loss, "logits": logits}


class Qwen3_5MoeStageModule(nn.Module):
    """One pipeline-parallel stage of the Qwen3.5-MoE text backbone.

    Holds a contiguous slab of decoder layers plus, on the boundary stages,
    ``embed_tokens`` (first) and ``norm`` + ``lm_head`` (last). The forward uses
    the same fp32 residual stream as :meth:`Qwen3_5MoeForCausalLM.forward`; the
    last stage returns a **sum-reduced** cross-entropy on pre-shifted targets so
    the pipeline schedule's per-micro-batch grads are token-summed (the trainer
    divides by the global valid-token count to recover the token-mean).

    Args:
        layers: This stage's decoder layers.
        embed_tokens: Token embedding (first stage only, else ``None``).
        norm: Final RMSNorm (last stage only, else ``None``).
        lm_head: Output projection (last stage only, else ``None``).
    """

    def __init__(self, layers, embed_tokens=None, norm=None, lm_head=None):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Run this stage; last stage returns sum-CE (if ``targets``) else logits."""
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_states)
        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        position_ids = _normalize_qwen3_5_position_ids(
            position_ids, bsz, seq_len, hidden_states.device,
        )
        linear_attention_mask, causal_attention_mask = _prepare_qwen3_5_attention_masks(
            attention_mask, seq_len, hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=causal_attention_mask,
                linear_attention_mask=linear_attention_mask,
            )
        if self.norm is None or self.lm_head is None:
            return hidden_states
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        if targets is None:
            return logits
        logits_fp = logits.float()
        return F.cross_entropy(
            logits_fp.view(-1, logits_fp.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
