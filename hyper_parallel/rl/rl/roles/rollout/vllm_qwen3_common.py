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
"""Qwen3 dense/MoE helpers shared by Hyper-vLLM model adapters."""

from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention


def join_prefix(prefix: str, suffix: str) -> str:
    """Join one optional vLLM module prefix."""
    return f"{prefix}.{suffix}" if prefix else suffix


def config_value(config: object, name: str, default: Any = None) -> Any:
    """Read a config value while treating ``None`` as the default."""
    value = getattr(config, name, default)
    return default if value is None else value


def normalize_positions(
    positions: torch.Tensor,
    num_tokens: int,
    *,
    family: str,
) -> torch.Tensor:
    """Normalize vLLM packed-token positions for a Transformers decoder."""
    if positions.ndim == 1 and positions.shape[0] == num_tokens:
        return positions.unsqueeze(0)
    if positions.ndim == 2 and positions.shape == (1, num_tokens):
        return positions
    raise ValueError(f"{family} positions must have shape [T] or [1,T]")


class Qwen3PagedAttention(nn.Module):
    """Keep Transformers Qwen3 projections around vLLM paged attention.

    Dense Qwen3 and Qwen3-MoE expose the same projection, QK-normalization,
    RoPE, and head metadata contract.  The only vLLM-owned compute here is
    the attention leaf because it owns paged KV cache and decode scheduling.
    """

    def __init__(
        self,
        attention: nn.Module,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        family: str,
    ) -> None:
        """Replace one Qwen3 attention compute leaf with vLLM attention."""
        super().__init__()
        self.family = family
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.head_dim = attention.head_dim
        self.num_heads = attention.config.num_attention_heads
        self.num_key_value_heads = attention.config.num_key_value_heads
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        if self.num_heads % tp_size or self.num_key_value_heads % tp_size:
            raise ValueError(
                "Qwen3 attention heads must divide tensor_parallel_size: "
                f"heads={self.num_heads}, kv_heads={self.num_key_value_heads}, tp={tp_size}"
            )
        self.scaling = attention.scaling
        self.layer_idx = attention.layer_idx
        self.attention = Attention(
            num_heads=self.num_heads // tp_size,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_key_value_heads // tp_size,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=attention.sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        """Run Transformers projections and RoPE with vLLM-owned KV state."""
        del kwargs
        if attention_mask is not None:
            raise ValueError("vLLM manages causal masks; explicit attention_mask is unsupported")
        if past_key_values is not None:
            raise ValueError("vLLM owns KV cache state; Transformers past_key_values is unsupported")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                f"{self.family} expects packed hidden states with shape [1,T,H]"
            )

        batch_size, num_tokens, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query = query.transpose(1, 2).reshape(num_tokens, -1)
        key = key.transpose(1, 2).reshape(num_tokens, -1)
        value = value.transpose(1, 2).reshape(num_tokens, -1)
        output = self.attention(query, key, value)
        output = output.view(batch_size, num_tokens, -1)
        return self.o_proj(output), None


__all__ = [
    "Qwen3PagedAttention",
    "config_value",
    "join_prefix",
    "normalize_positions",
]
