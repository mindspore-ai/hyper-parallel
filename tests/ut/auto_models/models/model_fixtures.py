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
"""Tiny model family fixtures for ``tests/ut/auto_models/models``.

The modules below mimic the Llama/MoE/Qwen family shapes used by planner and
applier tests: stable FQNs (``model.layers.{i}.self_attn.q_proj.weight`` ...),
predictable parameter identity, tied embeddings, and cached head-count
attributes — without importing Transformers.
"""

from types import SimpleNamespace

import torch
from torch import nn


class TinyAttention(nn.Module):
    """HF-style attention with cached head counts and q/k/v/o projections."""

    def __init__(self, hidden_size=8, num_heads=2, num_kv_heads=1):
        super().__init__()
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.o_proj(self.q_proj(hidden_states))


class TinyMLP(nn.Module):
    """HF-style gated MLP (gate/up/down)."""

    def __init__(self, hidden_size=8, intermediate_size=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class TinyDecoderLayer(nn.Module):
    """HF-style decoder layer with self_attn/mlp and both norms."""

    def __init__(self, hidden_size=8, num_heads=2, num_kv_heads=1, intermediate_size=16):
        super().__init__()
        self.self_attn = TinyAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = TinyMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TinyInnerModel(nn.Module):
    """The ``model`` submodule: embed + layers + final norm."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyDecoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.intermediate_size,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class TinyCausalLM(nn.Module):
    """HF-style ``*ForCausalLM``: ``model`` + ``lm_head``, optionally tied."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TinyInnerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))


def tiny_llama_config(**overrides):
    """A minimal Llama-like config namespace with stable small dimensions."""
    values = {
        "model_type": "llama",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_llama_model(**overrides):
    """Build a deterministic TinyCausalLM (fixed seed => stable parameters)."""
    torch.manual_seed(0)
    return TinyCausalLM(tiny_llama_config(**overrides))


class TinyMoEMLP(nn.Module):
    """MoE block: router + per-expert gated MLPs in a ``ModuleList``."""

    def __init__(self, hidden_size=8, intermediate_size=16, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        )

    def forward(self, hidden_states):
        weights = torch.softmax(self.gate(hidden_states), dim=-1)
        output = torch.zeros_like(hidden_states)
        for index, expert in enumerate(self.experts):
            output = output + weights[..., index : index + 1] * expert(hidden_states)
        return output


class TinyMoEDecoderLayer(TinyDecoderLayer):
    """Decoder layer whose MLP is a MoE block (FQN ``...layers.{i}.mlp.experts``)."""

    def __init__(self, hidden_size=8, num_heads=2, num_kv_heads=1,
                 intermediate_size=16, num_experts=2):
        super().__init__(hidden_size, num_heads, num_kv_heads, intermediate_size)
        self.mlp = TinyMoEMLP(hidden_size, intermediate_size, num_experts)


def tiny_moe_config(**overrides):
    """A minimal MoE config namespace (Qwen-MoE-shaped FQNs)."""
    values = {
        "model_type": "qwen3_moe",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts": 2,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_moe_model(**overrides):
    """Build a deterministic MoE TinyCausalLM variant (fixed seed)."""
    config = tiny_moe_config(**overrides)
    torch.manual_seed(0)
    model = TinyCausalLM(config)
    model.model.layers = nn.ModuleList(
        TinyMoEDecoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.intermediate_size,
            config.num_experts,
        )
        for _ in range(config.num_hidden_layers)
    )
    return model
