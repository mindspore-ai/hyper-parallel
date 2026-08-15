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
"""Tests for Qwen3-30B-A3B attention activation swapping."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
)

from hyper_models.components.activation_swap.attention_swap import (
    apply_qwen3_moe_attention_swap,
    qwen3_attention_swap_policy,
    validate_attention_swap,
)
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy
from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import SwapWrapper


def _new_module(module_type: type[nn.Module]) -> nn.Module:
    """Construct a module instance without allocating its production weights."""
    module = module_type.__new__(module_type)
    nn.Module.__init__(module)
    return module


def _qwen3_30b_a3b_stub() -> Qwen3MoeForCausalLM:
    """Build the supported HF module hierarchy with lightweight layer contents."""
    model = _new_module(Qwen3MoeForCausalLM)
    model.config = SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        head_dim=128,
        hidden_size=2048,
        intermediate_size=6144,
        max_position_embeddings=40960,
        moe_intermediate_size=768,
        num_attention_heads=32,
        num_experts=128,
        num_experts_per_tok=8,
        num_hidden_layers=48,
        num_key_value_heads=4,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        use_sliding_window=False,
        vocab_size=151936,
    )
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    for _ in range(model.config.num_hidden_layers):
        layer = _new_module(Qwen3MoeDecoderLayer)
        attention = _new_module(Qwen3MoeAttention)
        attention.proj = nn.Linear(2, 2)
        layer.self_attn = attention
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = nn.Linear(2, 2)
        model.model.layers.append(layer)
    return model


def test_attention_swap_policy_filters_unsafe_or_small_tensors() -> None:
    no_grad = torch.empty(1024, 1024)
    one_dimensional = torch.empty(1024 * 1024, requires_grad=True)
    small = torch.empty(16, 16, requires_grad=True)
    base = torch.empty(1024, 1024, requires_grad=True)
    shared_storage_view = base[:512]
    large = torch.empty(512, 512, requires_grad=True)

    assert qwen3_attention_swap_policy(no_grad) is CheckpointPolicy.MUST_SAVE
    assert qwen3_attention_swap_policy(one_dimensional) is CheckpointPolicy.MUST_SAVE
    assert qwen3_attention_swap_policy(small) is CheckpointPolicy.MUST_SAVE
    assert qwen3_attention_swap_policy(shared_storage_view) is CheckpointPolicy.MUST_SAVE
    assert qwen3_attention_swap_policy(large) is CheckpointPolicy.MUST_SWAP


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enable_compile": True}, "torch.compile"),
        ({"activation_checkpoint": "full"}, "activation checkpointing"),
        ({"pp_size": 2}, "pipeline parallelism"),
    ],
)
def test_attention_swap_rejects_incompatible_features(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_attention_swap("attention", **kwargs)


def test_attention_swap_only_wraps_attention_and_schedules_local_layers() -> None:
    model = _qwen3_30b_a3b_stub()
    state_dict_keys = set(model.state_dict())
    norms = [layer.input_layernorm for layer in model.model.layers]
    mlps = [layer.mlp for layer in model.model.layers]
    manager = MagicMock()

    with patch(
        "hyper_models.components.activation_swap.attention_swap.SwapManager",
        return_value=manager,
    ):
        result = apply_qwen3_moe_attention_swap(model, "attention")

    assert result is model
    assert all(isinstance(layer.self_attn, SwapWrapper) for layer in model.model.layers)
    assert [layer.input_layernorm for layer in model.model.layers] == norms
    assert [layer.mlp for layer in model.model.layers] == mlps
    assert set(model.state_dict()) == state_dict_keys
    assert manager.set_forward_prefetch_layer.call_count == 47
