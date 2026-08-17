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
"""Qwen3-VL-MoE conditional generation model.

Implements the ``Qwen3VLMoeForConditionalGeneration`` text + vision architecture
(vision tower, DeepStack visual injection, MoE text decoder).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.attention import GroupQueryAttention, _expand_kv_heads
from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.modules.rope import MultiModalRotaryEmbedding, apply_rotary_pos_emb
from hyper_parallel.models.qwen3_vl_vision import (
    Qwen3VLMoeVisionConfig,
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeVisionOutput,
)
from hyper_parallel.tools.logging import get_logger

# INFO via ``HP_LOG_CONFIG=HP:INFO`` to see the [v1-kernels] lines.
_hp_logger = get_logger("HP")


def _shifted_ce_loss(logits, labels):
    """Next-token cross-entropy loss from raw labels."""
    logits_fp = logits.float()
    targets = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    return F.cross_entropy(
        logits_fp.view(-1, logits_fp.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )


_V1_KERNELS_LOGGED = False
_V1_DISPATCH_LOGGED = False


def _use_v1_kernels() -> bool:
    """True when ``HYPER_USE_V1_KERNELS=1`` and ``torch_npu`` is importable.

    When set, the MoE experts dispatch to the fused ``npu_grouped_matmul``
    path instead of the eager per-expert loop. Logs the negative cases once;
    the positive "ON" line comes from :func:`_log_v1_dispatch_once` at the
    dispatch site, so it prints only when the fused branch actually executes.
    """
    global _V1_KERNELS_LOGGED
    requested = os.environ.get("HYPER_USE_V1_KERNELS", "0") == "1"
    available = False
    if requested:
        try:
            import torch_npu  # pylint: disable=C0415,W0611
            available = True
        except ImportError:
            available = False
    if not _V1_KERNELS_LOGGED:
        _V1_KERNELS_LOGGED = True
        if not requested:
            _hp_logger.info(
                "[v1-kernels] OFF (HYPER_USE_V1_KERNELS!=1) -- using the eager "
                "per-expert MoE path (slow backward).")
        elif not available:
            _hp_logger.warning(
                "[v1-kernels] REQUESTED (HYPER_USE_V1_KERNELS=1) but torch_npu "
                "import FAILED -- falling back to the eager per-expert MoE path "
                "(slow backward).")
        # requested AND available: stay quiet here; the dispatch announces "ON"
        # only when the fused branch is genuinely entered on NPU tensors.
    return requested and available


def _log_v1_dispatch_once(hidden_states: "torch.Tensor") -> None:
    """Emit the one-time ``[v1-kernels] ON`` line when the fused MoE branch runs.

    Downgrades to ``WARNING`` for non-bfloat16 inputs: the bf16-oriented fused
    NPU ops have no cast guard and may error or silently downcast.
    """
    global _V1_DISPATCH_LOGGED
    if _V1_DISPATCH_LOGGED:
        return
    _V1_DISPATCH_LOGGED = True
    dtype = hidden_states.dtype
    if dtype == torch.bfloat16:
        _hp_logger.info(
            "[v1-kernels] ON -- fused npu_grouped_matmul MoE path executing "
            "(dtype=%s).", dtype)
    else:
        _hp_logger.warning(
            "[v1-kernels] ON but dtype=%s (expected bfloat16) -- fp32/other is "
            "passed straight into the bf16-oriented fused NPU ops, which may "
            "error or silently downcast; use param_dtype=bfloat16 with v1 kernels.",
            dtype)


class _GmmFunction(torch.autograd.Function):
    """Custom autograd op around ``torch_npu.npu_grouped_matmul``.

    Inputs:
      - ``x``: (T_perm, K) — token-permuted input.
      - ``weight``: (E, K, N) — per-expert weight stack.
      - ``group_list``: (E,) int64 — token count per expert (cumulative semantics
        controlled via ``group_list_type=1`` = absolute counts).

    Forward returns (T_perm, N).
    """

    # pylint: disable=W0223  # intentional — only forward/backward needed
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                group_list: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
        # pylint: disable=C0415
        """Forward pass."""
        import torch_npu
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        out = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list,
            split_item=2, group_type=0, group_list_type=1,
        )[0]
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=C0415
        """Backward pass."""
        import torch_npu
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        weight_t = torch.transpose(weight, 1, 2)
        grad_x = torch_npu.npu_grouped_matmul(
            [grad_output], [weight_t], bias=None, group_list=group_list,
            split_item=2, group_type=0, group_list_type=1,
        )[0]
        grad_w = torch_npu.npu_grouped_matmul(
            [x.T], [grad_output], bias=None, group_list=group_list,
            split_item=3, group_type=2, group_list_type=1,
        )[0]
        return grad_x, grad_w, None


class _Qwen3VLMoeGroupedMM(torch.autograd.Function):
    """Per-expert grouped matmul used by Qwen3-VL-MoE packed experts."""

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
                grad_input[start:end].copy_(
                    torch.mm(grad_output[start:end], weight[expert_idx].T)
                )
                grad_weight[expert_idx].copy_(
                    torch.mm(
                        sorted_input[start:end].to(grad_weight.dtype).T,
                        grad_output[start:end].to(grad_weight.dtype),
                    )
                )
            start = end
        return grad_input, grad_weight, None


def _qwen3_vl_moe_grouped_mm(
    sorted_input: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3-VL-MoE grouped matmul on already expert-major tokens."""
    return _Qwen3VLMoeGroupedMM.apply(sorted_input, weight, offsets)


class Qwen3VLMoeRMSNorm(nn.Module):
    """Qwen3-VL-MoE RMSNorm with DTensor sequence placement preservation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_dtype = hidden_states.dtype
        hidden_states_fp = hidden_states.float()
        variance = hidden_states_fp.pow(2).mean(-1, keepdim=True)
        normed = hidden_states_fp * torch.rsqrt(variance + self.eps)
        return normed.to(input_dtype) * self.weight


@dataclass
class Qwen3VLMoeTextConfig:
    """Text config fields used by Qwen3-VL-MoE 30B-A3B."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 4
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    rope_theta: float = 5_000_000.0
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])

    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768

    # Production / industry default (matches MindFormers ``use_flash_attention``):
    # ``"flash_attention_2"`` routes the text decoder through the fused
    # ``torch_npu.npu_fusion_attention`` kernel (text head_dim 128 is
    # flash-supported). ``"eager"`` / ``"sdpa"`` use the shared SDPA path; switch
    # to ``"eager"`` only for strict eager-kernel debugging.
    _attn_implementation: str = "flash_attention_2"



@dataclass
class Qwen3VLMoeConfig:
    """Composite config for native Qwen3-VL-MoE conditional generation."""

    text_config: Qwen3VLMoeTextConfig = field(default_factory=Qwen3VLMoeTextConfig)
    vision_config: Qwen3VLMoeVisionConfig = field(default_factory=Qwen3VLMoeVisionConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vl: bool = True


class Qwen3VLMoeTextTopKRouter(nn.Module):
    """Router matching HF ``Qwen3VLMoeTextTopKRouter``."""

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):  # pylint: disable=W0613
        # Routing: bf16 ``F.linear`` → fp32 softmax → ``torch.topk``. The
        # ``topk`` is deterministic under ``use_deterministic_algorithms``.
        """Forward pass."""
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        router_top_value, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1,
        )

        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(hidden_states.dtype)
        return router_logits, router_top_value, router_indices


class Qwen3VLMoeTextExperts(nn.Module):
    """Packed expert weights for Qwen3-VL-MoE text experts.

    Parameters use the HF runtime layout: ``gate_up_proj`` is ``(E, 2I, H)``
    and ``down_proj`` is ``(E, H, I)``. The checkpoint stores the grouped-GEMM
    layout ``(E, H, 2I)`` / ``(E, I, H)``, so the state-dict adapter transposes
    once at load time. Eager forward uses the local grouped-mm fallback order.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
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
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        if _use_v1_kernels() and hidden_states.device.type == "npu":
            _log_v1_dispatch_once(hidden_states)
            return self._npu_v1_forward(hidden_states, routing_weights, router_indices)
        num_tokens, hidden_dim = hidden_states.shape
        num_top_k = router_indices.size(-1)
        device = hidden_states.device

        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1).expand(-1, num_top_k).reshape(-1)
        )
        sample_weights = routing_weights.reshape(-1)
        expert_ids = router_indices.reshape(-1)

        invalid_mask = expert_ids >= self.num_experts
        expert_ids = expert_ids.clamp(0, self.num_experts - 1)

        perm = torch.argsort(
            expert_ids,
            stable=getattr(self, "_hp_moe_stable_sort", False),
        )
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)

        expert_ids_sorted = expert_ids[perm]
        sample_weights_sorted = sample_weights[perm]
        sorted_hidden = hidden_states[token_idx[perm]]
        offsets = self._expert_offsets(expert_ids_sorted, self.num_experts)

        gate_up = _qwen3_vl_moe_grouped_mm(
            sorted_hidden, self.gate_up_proj.transpose(-2, -1), offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        down = _qwen3_vl_moe_grouped_mm(
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
        gate_up_out = _qwen3_vl_moe_grouped_mm(routed_input, gate_up.transpose(-2, -1), offsets)
        gate_part, up_part = gate_up_out.chunk(2, dim=-1)
        intermediate = F.silu(gate_part) * up_part
        return _qwen3_vl_moe_grouped_mm(intermediate, down.transpose(-2, -1), offsets)

    def _npu_v1_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        """NPU fused MoE expert path (``npu_moe_token_permute`` +
        ``npu_grouped_matmul`` + ``npu_swiglu`` + ``npu_moe_token_unpermute``).

        Expects sparse ``routing_weights`` (num_tokens, top_k); the sparse
        form is supplied by the sparse MoE block when
        ``HYPER_USE_V1_KERNELS=1``.
        """
        # pylint: disable=C0415
        import torch_npu
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
            hidden_states, router_indices.to(torch.int32),
        )
        tokens_per_expert = torch.histc(
            router_indices, bins=self.num_experts, min=0, max=self.num_experts,
        ).to(torch.int64)
        gate_up = self.gate_up_proj.transpose(1, 2).contiguous()
        down = self.down_proj.transpose(1, 2).contiguous()
        intermediate_hidden_states = _GmmFunction.apply(
            permuted_hidden_states, gate_up, tokens_per_expert,
        )
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = _GmmFunction.apply(
            intermediate_activations, down, tokens_per_expert,
        )
        next_states = torch_npu.npu_moe_token_unpermute(

            output, row_ids_map, probs=routing_weights,
        )
        return next_states


class Qwen3VLMoeTextSparseMoE(nn.Module):
    """Sparse MoE for the text decoder.

    Forward:
        router_logits = self.gate(hidden_states)              # bf16
        routing_weights = softmax(router_logits, dtype=fp32)
        routing_weights, router_indices = topk(routing_weights, k)
        routing_weights /= routing_weights.sum(-1, kd)
        routing_weights = routing_weights.to(hidden_states.dtype)
        routed_out = self.experts(hidden_states, router_indices, routing_weights)

    Returns ``(routed_out, router_logits)`` as a tuple so the decoder
    callsite can unpack via ``isinstance(out, tuple)``.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = Qwen3VLMoeTextTopKRouter(config)
        self.experts = Qwen3VLMoeTextExperts(config)

    def forward(self, hidden_states: torch.Tensor):  # pylint: disable=W0613
        """Forward pass."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)
        router_logits, routing_weights, router_indices = self.gate(hidden_states_2d)
        del router_logits
        if _use_v1_kernels() and hidden_states.device.type == "npu":
            _log_v1_dispatch_once(hidden_states)
            # NPU path: pass sparse (T, top_k) routing_weights — consumed
            # directly by ``npu_moe_token_unpermute(probs=...)``.
            next_states = self.experts(
                hidden_states_2d, router_indices, routing_weights,
            )
            return next_states.reshape(batch_size, sequence_length, hidden_dim)

        next_states = self.experts(
            hidden_states_2d, router_indices, routing_weights,
        )
        return next_states.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3VLMoeTextSdpaCore(nn.Module):
    """Causal SDPA core for Qwen3-VL-MoE text attention."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """Run causal SDPA on ``[B, H, S, D]`` Q/K/V."""
        sdpa_kwargs = {"enable_gqa": True} if enable_gqa else {}
        if attention_mask is not None:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=False, scale=scale, **sdpa_kwargs,
            )
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, **sdpa_kwargs)


class Qwen3VLMoeTextAttention(GroupQueryAttention):
    """Qwen3-VL-MoE text attention with selectable attention backend.

    Subclasses :class:`GroupQueryAttention` for projection layout without
    modifying the shared module; overrides ``forward`` to dispatch to SDPA or
    flash_attention_2 when requested.
    """

    def __init__(self, attn_implementation: str = "eager", **kwargs):
        super().__init__(**kwargs)
        self._attn_implementation = attn_implementation
        # ``eager_attention_forward`` (delegated to via the eager path) reads
        # ``module.num_key_value_groups``; alias to the local field.
        self.num_key_value_groups = self.num_kv_groups
        self.sdpa_core = Qwen3VLMoeTextSdpaCore()

    def forward(self, hidden_states: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None, **kwargs):  # pylint: disable=W0613
        """Dispatch attention computation based on the configured implementation."""
        # eager / sdpa / fa2: replicate projection + rotary explicitly so the
        # selected kernel (eager fp32-softmax matmul, or NPU fusion attention)
        # is exercised directly rather than the parent's SDPA path.
        bsz, seq_len, _ = hidden_states.shape
        q_out = self.q_proj(hidden_states)
        if q_out.shape[-1] % self.head_dim != 0:
            raise ValueError(
                f"q_proj output dim {q_out.shape[-1]} is not divisible by head_dim {self.head_dim}."
            )
        q_raw = q_out.reshape(bsz, seq_len, -1, self.head_dim)
        q = self.q_norm(q_raw).transpose(1, 2)
        k_out = self.k_proj(hidden_states)
        if k_out.shape[-1] % self.head_dim != 0:
            raise ValueError(
                f"k_proj output dim {k_out.shape[-1]} is not divisible by head_dim {self.head_dim}."
            )
        k = self.k_norm(
            k_out.reshape(bsz, seq_len, -1, self.head_dim)
        ).transpose(1, 2)
        v_out = self.v_proj(hidden_states)
        if v_out.shape[-1] != k_out.shape[-1]:
            raise ValueError(
                f"v_proj output dim {v_out.shape[-1]} must match k_proj output dim {k_out.shape[-1]}."
            )
        v = v_out.reshape(bsz, seq_len, -1, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = kwargs.get("attention_mask")
        scaling = self.head_dim ** -0.5

        if self._attn_implementation == "sdpa":
            kv_groups = q.shape[1] // k.shape[1]
            expand_kv_for_cp = bool(getattr(self, "_hp_cp_expand_kv_before_core", False))
            enable_gqa = kv_groups > 1 and attn_mask is None and not expand_kv_for_cp
            if kv_groups > 1 and (expand_kv_for_cp or not enable_gqa):
                k = _expand_kv_heads(k, kv_groups)
                v = _expand_kv_heads(v, kv_groups)
            attn_output = self.sdpa_core(
                q, k, v, attention_mask=attn_mask, scale=scaling, enable_gqa=enable_gqa,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            return self.o_proj(attn_output)

        if self._attn_implementation == "flash_attention_2":
            # pylint: disable=C0415
            from transformers.modeling_flash_attention_utils import _flash_attention_forward
            # fa2 only consumes the 2D padding form + ``is_causal``.
            if attn_mask is not None and attn_mask.ndim == 4:
                attn_mask = None
            attn_output = _flash_attention_forward(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attention_mask=attn_mask,
                query_length=seq_len,
                is_causal=True,
                dropout=0.0,
                softmax_scale=scaling,
                attn_implementation="flash_attention_2",
            )
            attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
            return self.o_proj(attn_output)

        # eager: delegate so the NPU kernel-cache key stays stable.
        # pylint: disable=C0415
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            eager_attention_forward as _eager,
        )
        attn_output, _ = _eager(
            self, q, k, v,
            attention_mask=attn_mask,
            scaling=scaling,

            dropout=0.0,
        )
        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen3VLMoeTextDecoder(nn.Module):
    """One Qwen3-VL-MoE text decoder layer."""

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        layer_idx: int,
        rope: MultiModalRotaryEmbedding,
    ):
        super().__init__()
        # Pure-GQA: every VL text layer is a full-attention layer.
        self.layer_type = "full_attention"
        self.self_attn = Qwen3VLMoeTextAttention(
            attn_implementation=getattr(config, "_attn_implementation", "eager"),
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_bias=config.attention_bias,
            out_bias=config.attention_bias,
            rope=rope,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            norm_cls=Qwen3VLMoeRMSNorm,
        )
        if (
            layer_idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3VLMoeTextSparseMoE(config)
        else:
            self.mlp = SwiGLUMLP(
                config.hidden_size, config.intermediate_size, bias=False,
            )
        self.input_layernorm = Qwen3VLMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3VLMoeRMSNorm(
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
        hidden_states = self.self_attn(
            hidden_states, position_ids=position_ids, attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # ``Qwen3VLMoeTextSparseMoE`` returns either a plain tensor or a
        # ``(routed_out, router_logits)`` tuple depending on whether

        # ``output_router_logits`` is enabled — unpack the tuple form here.
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        return residual + hidden_states


class Qwen3VLMoeTextModel(nn.Module):
    """Text decoder used by the multimodal conditional generation wrapper."""

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = MultiModalRotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Per-layer rotary instances (one non-persistent inv_freq buffer
        # each): a single shared module registered under every layer makes
        # the activation-checkpoint wrapper see overlapping wrap regions.
        self.layers = nn.ModuleList([
            Qwen3VLMoeTextDecoder(config, i, MultiModalRotaryEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
                mrope_section=config.mrope_section,
            ))
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3VLMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_input = nn.Identity()
        self.deepstack_input = nn.Identity()
        self.deepstack_output = nn.Identity()

    @staticmethod
    def _build_causal_mask(
        attention_mask: Optional[torch.Tensor],
        bsz: int,
        seq_len: int,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Build a 4D additive causal+padding mask matching HF create_causal_mask.

        HF ``create_causal_mask`` produces a ``[B, 1, S, S]`` float mask where
        attended positions are 0.0 and blocked positions are ``-inf`` (dtype min).
        Rules:
        (1) causal: upper-triangle is blocked (future tokens cannot be attended)
        (2) padding: columns for pad tokens (``attention_mask==0``) are -inf so
            no query can attend to a pad key.

        Returns a float tensor of shape ``[B, 1, S, S]``.
        """
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_val = torch.finfo(dtype).min

        # [S, S] causal upper-triangle: -inf where blocked
        causal = torch.triu(
            torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device),
            diagonal=1,
        )
        mask_4d = causal.view(1, 1, seq_len, seq_len).expand(bsz, 1, seq_len, seq_len).clone()

        # Apply padding column mask when a 1D/2D attention_mask is provided.
        if attention_mask is not None and attention_mask.ndim <= 2:
            # attention_mask: [B, S] with 1=attend, 0=pad → block pad columns
            pad_cols = attention_mask == 0  # [B, S], True where padding
            mask_4d = mask_4d.masked_fill(pad_cols.view(bsz, 1, 1, seq_len), min_val)

        return mask_4d

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Deepstack process (internal)."""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=inputs_embeds.device, dtype=torch.long,
            ).view(1, -1).expand(bsz, -1)
        if position_ids.ndim == 4:
            position_ids = position_ids.squeeze(0)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        # fa2 expects the raw 2D padding mask (it combines causal + padding
        # internally); SDPA paths get a pre-built 4D causal+padding mask.
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if attn_impl == "flash_attention_2":
            layer_attention_mask = attention_mask
        elif (
            attn_impl == "sdpa"
            and (attention_mask is None or (attention_mask.ndim <= 2 and torch.all(attention_mask == 1)))
        ):
            layer_attention_mask = None
        else:
            layer_attention_mask = self._build_causal_mask(
                attention_mask, bsz, seq_len, inputs_embeds,
            )

        inputs_embeds, position_ids = self.decoder_input((inputs_embeds, position_ids))

        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=layer_attention_mask,
            )
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                hidden_states = self.deepstack_input(hidden_states)
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx],
                )
                hidden_states = self.deepstack_output(hidden_states)
        return self.norm(hidden_states)


class Qwen3VLMoeModel(nn.Module):
    """Composite Qwen3-VL-MoE model with native visual token injection."""

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLMoeVisionModel(config.vision_config)
        self.language_model = Qwen3VLMoeTextModel(config.text_config)
        self.rope_deltas = None
        self.visual_injection_input = nn.Identity()
        self.visual_injection_output = nn.Identity()

    @property
    def layers(self):
        """Return the language model decoder layer list."""
        return self.language_model.layers

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text model's token embedding table."""
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        """Set the text token embedding."""
        self.language_model.embed_tokens = value


    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get vision position ids."""
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        position_height = torch.arange(llm_grid_h, device=device) + start_position

        position_width = position_width.repeat(llm_grid_h * llm_grid_t)
        position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
        position_temporal = (
            position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
        )
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get rope index."""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        rope_deltas = []
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        for batch_idx in range(input_ids.shape[0]):
            token_type = mm_token_type_ids[batch_idx]
            current_input_ids = input_ids[batch_idx]
            if attention_mask is not None:
                mask = attention_mask[batch_idx].bool()
                token_type = token_type[mask]
                current_input_ids = current_input_ids[mask]
            groups = []
            prev = None
            start = 0
            for idx, val in enumerate(token_type.tolist()):
                if prev is None:
                    prev = val
                    start = idx
                elif val != prev:
                    groups.append((prev, start, idx))
                    prev = val
                    start = idx
            if prev is not None:
                groups.append((prev, start, len(token_type)))

            current_pos = 0
            pos_parts = []
            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    pos_parts.append(
                        torch.arange(
                            text_len, device=input_ids.device,
                        ).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid = next(grid_iters[modality_type])
                    pos_parts.append(
                        self.get_vision_position_ids(
                            current_pos, grid, 1, spatial_merge_size,
                            device=input_ids.device,
                        )
                    )
                    current_pos += max(grid[1], grid[2]).item() // spatial_merge_size
            llm_positions = torch.cat(pos_parts, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions
            rope_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        return position_ids, torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Qwen3VLMoeVisionOutput:
        """Get image features."""
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.visual.spatial_merge_size ** 2)
        ).tolist()
        vision_output.pooler_output = torch.split(
            vision_output.pooler_output, split_sizes,
        )
        return vision_output

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get placeholder mask.

        Computed from ``input_ids`` and the hidden size, matching the single-card
        path without relying on embedding boolean-indexing side effects.
        """
        hidden_size = inputs_embeds.shape[-1]
        special_image_mask = input_ids == self.config.image_token_id
        n_image_tokens = int(special_image_mask.sum())
        if image_features is not None and n_image_tokens * hidden_size != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={tuple(image_features.shape)}"
            )
        return special_image_mask.unsqueeze(-1).expand(-1, -1, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            inputs_embeds = self.visual_injection_input(inputs_embeds)
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype,
            )
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = self.visual_injection_output(inputs_embeds)
            deepstack_visual_embeds = image_outputs.deepstack_features

        visual_pos_masks = image_mask[..., 0] if image_mask is not None else None

        if position_ids is None and image_grid_thw is not None:
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                mm_token_type_ids[input_ids == self.config.image_token_id] = 1
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )

        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,

            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )


class Qwen3VLMoeForCausalLM(nn.Module):
    """Text-only CausalLM wrapper with HF-compatible state-dict names."""

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = MultiModalRotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Per-layer rotary instances (one non-persistent inv_freq buffer
        # each): a single shared module registered under every layer makes
        # the activation-checkpoint wrapper see overlapping wrap regions.
        self.layers = nn.ModuleList([
            Qwen3VLMoeTextDecoder(config, i, MultiModalRotaryEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
                mrope_section=config.mrope_section,
            ))
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3VLMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

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
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device, dtype=torch.long,
            ).view(1, -1).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            gather_logits = getattr(self, "_hp_tp_logits_gather", None)
            if gather_logits is not None:
                logits = gather_logits(logits, labels)
            loss = _shifted_ce_loss(logits, labels)
        return {"loss": loss, "logits": logits}


class Qwen3VLMoeForConditionalGeneration(nn.Module):
    """Native multimodal Qwen3-VL-MoE conditional generation model."""

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3VLMoeModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    @property
    def layers(self):
        """Return the language model decoder layer list."""
        return self.model.language_model.layers

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        loss = None
        if labels is not None:
            gather_logits = getattr(self, "_hp_tp_logits_gather", None)
            if gather_logits is not None:
                logits = gather_logits(logits, labels)
            loss = _shifted_ce_loss(logits, labels)
        return {"loss": loss, "logits": logits}


class Qwen3VLMoeVisualPreprocess(nn.Module):
    """MPipe-Transpose preprocess: run the (frozen) visual tower and return its
    payload as a flat tuple ``(image_embeds, *deepstack_features)`` for the
    transpose transport.

    Duck-types :meth:`Qwen3VLMoeModel.get_image_features` (only ``self.visual``
    is needed). Holds the visual tower so it is present and broadcast-free on
    every rank (the tower is frozen, so MPipe ships only its output).
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, *args, pixel_values=None, image_grid_thw=None, **kwargs):  # pylint: disable=unused-argument
        """Encode this rank's images into the transposed visual payload."""
        image_outputs = Qwen3VLMoeModel.get_image_features(self, pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0)
        return (image_embeds, *image_outputs.deepstack_features)


class Qwen3VLMoeIdentityPreprocess(nn.Module):
    """Dataload-only MPipe preprocess: returns ``input_ids`` unchanged (the
    visual tower stays on stage 0). The param-free, ship-only baseline."""

    def forward(self, input_ids, **kwargs):  # pylint: disable=unused-argument
        """Pass the raw inputs through (dataload-only transpose, T=0)."""
        return input_ids


class Qwen3VLMoeStageModule(nn.Module):
    """One pipeline-parallel stage of Qwen3-VL-MoE.

    Stage 0 holds the visual tower + ``embed_tokens`` and runs the full
    visual-injection + 3D-mrope position-id computation (mirroring
    :meth:`Qwen3VLMoeModel.forward`); the last stage holds ``norm`` + ``lm_head``
    and returns a **sum-reduced** cross-entropy on pre-shifted targets (the
    trainer divides by the global valid-token count). The base cross-stage
    activations are ``(hidden_states, position_ids)`` — ``position_ids`` is the
    3D mrope tensor computed on stage 0 and carried forward (it cannot be a
    batch-split kwarg).

    DeepStack visual features inject after global layers ``< len(deepstack)``.
    When the layer split places injection layers beyond a stage's slab, the
    stage relays ``(visual_pos_masks, *remaining_features)`` to its successor as
    extra pipeline outputs: the mask travels as ``uint8`` (P2P rejects bool) and
    each feature tensor keeps its own ``requires_grad`` metadata, so a frozen
    tower relays pure data while a trainable tower's feature grads flow back
    hop-by-hop through the regular P2P grad path. The relayed tuple always
    covers global layers ``[layer_start, deepstack_len)`` of the *receiving*
    stage, shrinking at each boundary; stages past the last injection layer
    exchange the plain 2-tuple. The relay arity is part of the stage-pair P2P
    meta, so every micro-batch of a step must agree on whether images are
    present (the same sample-uniform layout PP micro-batching already requires).

    To reuse the composite model's vision helpers without duplicating the decoder
    layers, stage 0 holds ``visual`` + ``config`` and calls the unbound
    :class:`Qwen3VLMoeModel` helpers with ``self`` (duck typing).

    Args:
        layers: This stage's decoder layers.
        layer_start: Global index of ``layers[0]`` (for DeepStack mapping).
        config: Model config (``image_token_id`` / vision config) — stage 0 only.
        visual: Vision tower (stage 0 only, else ``None``).
        embed_tokens: Token embedding (stage 0 only, else ``None``).
        norm: Final RMSNorm (last stage only, else ``None``).
        lm_head: Output projection (last stage only, else ``None``).
        deepstack_len: Number of DeepStack features (injected after global layers
            ``0 .. deepstack_len-1``).
    """

    def __init__(
        self, layers, layer_start, config=None, visual=None,
        embed_tokens=None, norm=None, lm_head=None, deepstack_len=0,
        attn_impl="eager",
    ):
        super().__init__()
        self.layer_start = layer_start
        self.deepstack_len = deepstack_len
        self.attn_impl = attn_impl
        self.config = config
        self.visual = visual
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def get_vision_position_ids(self, *args, **kwargs):
        """Delegate to the composite helper (it uses no decoder-layer state)."""
        return Qwen3VLMoeModel.get_vision_position_ids(self, *args, **kwargs)

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        *deepstack_embeds_in: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        mpipe_visual: Optional[tuple] = None,
    ):
        """Run this stage.

        Non-last stages return ``(hidden_states, position_ids)`` plus, while
        injection layers remain downstream, ``(visual_pos_masks_u8,
        *remaining_deepstack_features)``; the last stage returns sum-CE (if
        ``targets``) else logits. ``deepstack_embeds_in`` (positional, from the
        predecessor's relay) covers global layers ``[layer_start,
        deepstack_len)``.

        ``mpipe_visual`` carries a MPipe-Transpose precomputed visual payload
        ``(image_embeds, *deepstack_features)`` (the frozen visual tower run on
        another rank). When provided, stage 0 injects it instead of running its
        own ``get_image_features`` — the 3D-mrope position-ids are still computed
        locally from ``image_grid_thw``.
        """
        # Per-stage view of the DeepStack features: ``deepstack_embeds[i]``
        # injects after global layer ``layer_start + i``.
        deepstack_embeds = list(deepstack_embeds_in) or None
        if self.embed_tokens is not None:  # stage 0: ``hidden_states`` is ``input_ids``
            input_ids = hidden_states
            inputs_embeds = self.embed_tokens(input_ids)
            if mpipe_visual is not None:
                image_embeds = mpipe_visual[0].to(inputs_embeds.device, inputs_embeds.dtype)
                deepstack_embeds = list(mpipe_visual[1:]) or None
                image_mask = Qwen3VLMoeModel.get_placeholder_mask(
                    self, input_ids, inputs_embeds, image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                # uint8 from the start: P2P meta rejects bool, and keeping one
                # dtype on every relay hop lets later stages forward it as-is.
                visual_pos_masks = image_mask[..., 0].to(torch.uint8)
            elif pixel_values is not None:
                # The two are split independently on dim 0, which is only
                # sample-aligned for a uniform batch; fail fast on ragged ones.
                expected_rows = int(image_grid_thw.prod(-1).sum())
                if expected_rows != pixel_values.shape[0]:
                    raise NotImplementedError(
                        "Qwen3-VL-MoE PP micro-batching split pixel_values "
                        f"({pixel_values.shape[0]} rows) and image_grid_thw "
                        f"({expected_rows} grid rows) onto mismatched boundaries. "
                        "pp_micro_batch_num>1 requires a sample-uniform VL batch "
                        "(every sample contributing an equal, dim-0-aligned number "
                        "of image patches); ragged per-sample image layouts are "
                        "not supported. Use pp_micro_batch_num=1, or pad the batch "
                        "so every sample has the same image/patch count."
                    )
                image_outputs = Qwen3VLMoeModel.get_image_features(
                    self, pixel_values, image_grid_thw,
                )
                image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype,
                )
                image_mask = Qwen3VLMoeModel.get_placeholder_mask(
                    self, input_ids, inputs_embeds, image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                deepstack_embeds = list(image_outputs.deepstack_features) or None
                # uint8 from the start: P2P meta rejects bool, and keeping one
                # dtype on every hop lets later stages relay the buffer as-is.
                visual_pos_masks = image_mask[..., 0].to(torch.uint8)
            if position_ids is None and image_grid_thw is not None:
                if mm_token_type_ids is None:
                    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                    mm_token_type_ids[input_ids == self.config.image_token_id] = 1
                position_ids, _ = Qwen3VLMoeModel.get_rope_index(
                    self, input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw, attention_mask=attention_mask,
                )
            if position_ids is None:
                # Image-free micro-batch: mRoPE collapses to plain 1D RoPE. Never emit a ``None``
                # activation; it crosses the stage boundary and the send path needs a tensor.
                bsz_, seq_ = input_ids.shape
                position_ids = torch.arange(
                    seq_, device=input_ids.device, dtype=torch.long,
                ).view(1, -1).expand(bsz_, -1)
            hidden_states = inputs_embeds

        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        # fa2 consumes the raw 2D padding mask (it combines causal + padding in
        # the kernel and drops any 4D mask); eager / sdpa get a pre-built 4D
        # causal+padding mask. Mirror ``Qwen3VLMoeTextModel.forward`` so a padded
        # flash-attention PP batch masks pad tokens correctly.
        if self.attn_impl == "flash_attention_2":
            layer_mask = attention_mask
        else:
            layer_mask = Qwen3VLMoeTextModel._build_causal_mask(  # pylint: disable=W0212
                attention_mask, bsz, seq_len, hidden_states,
            )
        # uint8 mask -> bool for local indexing; the uint8 buffer is kept intact so
        # it can be relayed to downstream stages unchanged.
        inject_mask = (
            visual_pos_masks.to(torch.bool) if visual_pos_masks is not None else None
        )
        for local_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, position_ids=position_ids, attention_mask=layer_mask,
            )
            global_idx = self.layer_start + local_idx
            if deepstack_embeds is not None and global_idx < self.deepstack_len:
                visual_embeds = deepstack_embeds[global_idx - self.layer_start].to(
                    hidden_states.device, hidden_states.dtype,
                )
                hidden_states = hidden_states.clone()
                hidden_states[inject_mask, :] = (
                    hidden_states[inject_mask, :] + visual_embeds
                )

        if self.norm is None or self.lm_head is None:
            next_start = self.layer_start + len(self.layers)
            if deepstack_embeds is not None and next_start < self.deepstack_len:
                # Injection layers remain downstream: relay the mask and the
                # features for global layers [next_start, deepstack_len).
                tail = deepstack_embeds[next_start - self.layer_start:]
                return (hidden_states, position_ids, visual_pos_masks, *tail)
            return hidden_states, position_ids
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
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
    "Qwen3VLMoeTextConfig",
    "Qwen3VLMoeVisionConfig",
    "Qwen3VLMoeConfig",
    "Qwen3VLMoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeStageModule",
    "Qwen3VLMoeTextDecoder",
]
