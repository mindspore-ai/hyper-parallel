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
"""Contract tests for the Transformers-based Hyper Qwen3 vLLM adapter."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

pytest.importorskip("vllm")

from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from vllm.config.compilation import CompilationMode

import rl.roles.rollout.vllm_qwen3 as qwen3_adapter
from rl.roles.rollout.vllm_qwen3 import (
    HyperQwen3ForCausalLM,
    _VLLMQwen3Attention,
    _validate_adapter_config,
)


class _FakeAttention(nn.Module):
    """Minimal vLLM attention leaf used by CPU adapter tests."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.impl = self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Return a correctly shaped deterministic attention placeholder."""
        del key, value
        return query


def _vllm_config(**parallel_overrides: int) -> SimpleNamespace:
    hf_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )
    parallel_config = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
    }
    parallel_config.update(parallel_overrides)
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=hf_config),
        parallel_config=SimpleNamespace(**parallel_config),
        cache_config=SimpleNamespace(),
        quant_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


def _build_model(monkeypatch: pytest.MonkeyPatch) -> HyperQwen3ForCausalLM:
    monkeypatch.setattr(qwen3_adapter, "Attention", _FakeAttention)
    return HyperQwen3ForCausalLM(vllm_config=_vllm_config())


def test_adapter_uses_transformers_model_with_vllm_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hyper keeps HF decoder modules and replaces only the stateful runtime leaf."""
    model = _build_model(monkeypatch)

    assert isinstance(model.model.layers[0].mlp, Qwen3MLP)
    assert isinstance(model.model.layers[0].self_attn, _VLLMQwen3Attention)
    assert isinstance(model.model.layers[0].self_attn.attention, _FakeAttention)
    assert model.get_input_embeddings() is model.model.embed_tokens

    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    positions = torch.arange(3, dtype=torch.long)
    hidden_states = model(input_ids, positions)

    assert hidden_states.shape == (3, model.config.hidden_size)
    assert model.compute_logits(hidden_states).shape == (3, model.config.vocab_size)


def test_adapter_strictly_loads_canonical_qwen3_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial load and online refit share the canonical Trainer namespace."""
    model = _build_model(monkeypatch)
    weights = [
        (name, torch.randn_like(parameter))
        for name, parameter in model.named_parameters()
    ]

    loaded = model.load_weights(iter(weights))

    assert loaded == {name for name, _ in model.named_parameters()}
    with pytest.raises(ValueError, match="missing parameters"):
        model.load_weights(iter(()))


def test_adapter_rejects_tensor_parallelism() -> None:
    """The first Hyper-vLLM delivery is explicitly TP1 only."""
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _validate_adapter_config(_vllm_config(tensor_parallel_size=2))
