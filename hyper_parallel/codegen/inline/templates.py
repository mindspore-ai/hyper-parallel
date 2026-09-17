# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Inline strategy templates rendered into generated modeling files."""

from __future__ import annotations


QWEN3_GQA_ATTENTION_CLASS = '''
class GQAAttention(nn.Module):
    """Generated Qwen3-MoE attention with visible TP/CP communication."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = getattr(config, "sliding_window", None)

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attention_interface = run_qwen3_moe_flash_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        actual_seq_len: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run fused GQA with generated TP/CP communication."""
        ps = get_parallel_state()
        if ps.tp_enabled:
            hidden_states = ps.tp.all_gather(hidden_states, dim=1)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        if ps.cp_enabled:
            cp_mesh = ps.cp_mesh
            query_length = query_states.shape[-2]
            query_offset = cp_mesh.get_local_rank() * query_length
            key_states, value_states = flex_cp_allgather(
                key_states.contiguous(), value_states.contiguous(), 2, cp_mesh
            )
            attention_mask = _cp_offset_causal_mask(
                query_length,
                key_states.shape[-2],
                query_offset,
                query_states.device,
            )

        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            actual_seq_len=actual_seq_len,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if ps.tp_enabled:
            attn_output = ps.tp.reduce_scatter(attn_output, dim=1)
        return attn_output, attn_weights
'''


PARALLEL_STATE_ACCESSOR = '''
def get_parallel_state():
    """Return the externally installed codegen parallel state."""
    return get_inline_parallel_state(__name__)
'''


TP_OPERATORS_CLASS = '''
def get_parallel_state():
    """Return the externally installed codegen parallel state."""
    return get_inline_parallel_state(__name__)


class TPOperators:
    """TP communication operators used directly by generated forwards."""

    def __init__(self, tp_group, tp_size, tp_rank, backend="hccl"):
        self._group = tp_group
        self._group_size = tp_size
        self._group_rank = tp_rank
        self._backend = str(backend).lower()

    def all_gather(self, tensor, dim=None):
        if tensor is None:
            return None
        platform = get_platform()
        return platform.differentiable_all_gather_concat(
            tensor, self._group, self._group_size, dim
        )

    def all_reduce(self, tensor):
        if tensor is None:
            return None
        platform = get_platform()
        return platform.differentiable_all_reduce(tensor, "sum", self._group)

    def reduce_scatter(self, tensor, dim=None):
        if tensor is None:
            return None
        platform = get_platform()
        if "gloo" in self._backend:
            reduced = platform.differentiable_all_reduce(tensor, "sum", self._group)
            return platform.chunk(reduced, dim, self._group_size, self._group_rank)
        return platform.differentiable_reduce_scatter(
            tensor, self._group_size, dim, "sum", self._group
        )
'''


MOE_EP_FORWARD_SHELL = '''"""Inline EP routed MoE forward (framework generic)."""
ps = get_parallel_state()
if not ps.ep_enabled:
    return self._forward_impl(hidden_states)

return moe_ep_forward(
    self,
    hidden_states,
    router_kind={router_kind!r},
    shared={shared!r},
    ep_group=ps.ep_group,
)
'''


def moe_ep_forward_body(router_kind: str, shared: str) -> str:
    """Return the thin inline EP forward body for the given structural keys.

    A single shared body referenced by every MoE render spec. The only
    inputs are the structure-selected router kind and shared merge mode, so
    no model-family EP dispatch literal remains in any adapter.
    """
    return MOE_EP_FORWARD_SHELL.format(router_kind=router_kind, shared=shared)
