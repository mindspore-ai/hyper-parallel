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
"""Contract tests for the Transformers-based Hyper Qwen3.5 vLLM adapter."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

pytest.importorskip("vllm")

from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM as TransformersQwen3_5ForCausalLM,
    Qwen3_5MLP,
    Qwen3_5RMSNormGated,
)
from vllm import ModelRegistry
from vllm.config.compilation import CompilationMode
from vllm.utils.torch_utils import set_default_torch_dtype

import rl.roles.rollout.vllm_qwen3_5 as qwen3_5_adapter
from rl.roles.rollout.vllm_plugin import (
    HYPER_QWEN3_5_ARCHITECTURE,
    register_hyper_models,
)
from rl.roles.rollout.vllm_qwen3_5 import (
    HyperQwen3_5ForCausalLM,
    _VLLMQwen3_5Attention,
    _VLLMQwen3_5GatedDeltaNet,
    _map_weight_name,
    _validate_adapter_config,
)


class _FakeAttention(nn.Module):
    """Minimal vLLM attention leaf used by CPU adapter tests."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Return a correctly shaped deterministic attention placeholder."""
        del key, value
        return query


class _FakeGdnRuntime(nn.Module):
    """Parameter-free stand-in for vLLM's request-state runtime."""

    def __init__(
        self,
        config: object,
        vllm_config: object,
        prefix: str,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del config, vllm_config, kwargs
        self.prefix = prefix


def _vllm_config(
    *,
    layer_types: list[str] | None = None,
    tensor_parallel_size: int = 1,
    tie_word_embeddings: bool = False,
) -> SimpleNamespace:
    if layer_types is None:
        layer_types = [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=tie_word_embeddings,
        attention_dropout=0.0,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=layer_types,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.75,
            "mrope_section": [1, 1, 1],
        },
    )
    text_config.mamba_ssm_dtype = "float32"
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        quant_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


def _build_model(
    monkeypatch: pytest.MonkeyPatch,
    **config_kwargs: object,
) -> HyperQwen3_5ForCausalLM:
    monkeypatch.setattr(qwen3_5_adapter, "Attention", _FakeAttention)
    monkeypatch.setattr(qwen3_5_adapter, "QwenGatedDeltaNetAttention", _FakeGdnRuntime)
    return HyperQwen3_5ForCausalLM(vllm_config=_vllm_config(**config_kwargs))


def test_registry_inspects_hybrid_text_model() -> None:
    """vLLM should discover generation and hybrid-state interfaces."""
    register_hyper_models()
    model_info = ModelRegistry.models[HYPER_QWEN3_5_ARCHITECTURE].inspect_model_cls()

    assert model_info.is_text_generation_model
    assert model_info.has_inner_state
    assert model_info.is_hybrid
    assert not model_info.supports_multimodal
    assert not model_info.supports_pp


def test_adapter_uses_transformers_modules_and_vllm_state_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only stateful attention and GDN execution should leave Transformers."""
    model = _build_model(monkeypatch)
    linear_layer = model.model.layers[0]
    attention_layer = model.model.layers[-1]

    assert isinstance(model, TransformersQwen3_5ForCausalLM)
    assert isinstance(linear_layer.mlp, Qwen3_5MLP)
    assert isinstance(linear_layer.linear_attn, _VLLMQwen3_5GatedDeltaNet)
    assert isinstance(linear_layer.linear_attn.norm, Qwen3_5RMSNormGated)
    assert isinstance(attention_layer.mlp, Qwen3_5MLP)
    assert isinstance(attention_layer.self_attn, _VLLMQwen3_5Attention)
    assert isinstance(attention_layer.self_attn.attention, _FakeAttention)
    assert model.get_input_embeddings() is model.model.embed_tokens
    assert not list(linear_layer.linear_attn.state_runtime.parameters())

    linear_layer.linear_attn.to_empty(device="meta")
    assert linear_layer.linear_attn.state_runtime.conv1d is linear_layer.linear_attn.conv1d
    assert linear_layer.linear_attn.state_runtime.dt_bias is linear_layer.linear_attn.dt_bias
    assert linear_layer.linear_attn.state_runtime.A_log is linear_layer.linear_attn.A_log


def test_full_attention_adapter_runs_packed_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF projections, partial MRoPE, output gate, and LM head should compose."""
    model = _build_model(monkeypatch, layer_types=["full_attention", "full_attention"])
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    positions = torch.arange(3, dtype=torch.long)

    hidden_states = model(input_ids, positions)

    assert hidden_states.shape == (3, model.config.hidden_size)
    assert model.compute_logits(hidden_states).shape == (3, model.config.vocab_size)


@pytest.mark.parametrize(
    "source_name,target_name",
    [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
        ),
        ("model.visual.patch_embed.proj.weight", None),
        ("model.language_model.rotary_emb.inv_freq", None),
    ],
)
def test_map_weight_name(source_name: str, target_name: str | None) -> None:
    """Composite text weights should map without admitting vision state."""
    assert _map_weight_name(source_name) == target_name


def test_adapter_strictly_loads_canonical_actor_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial load and online refit should share the canonical Actor namespace."""
    model = _build_model(monkeypatch)
    weights = [(name, torch.randn_like(parameter)) for name, parameter in model.named_parameters()]

    loaded = model.load_weights(iter(weights))

    assert loaded == {name for name, _ in model.named_parameters()}
    with pytest.raises(ValueError, match="missing parameters"):
        model.load_weights(iter(()))


def test_adapter_strictly_loads_composite_checkpoint_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composite checkpoint prefixes should resolve to the text-only HF model."""
    model = _build_model(monkeypatch)
    weights = []
    for name, parameter in model.named_parameters():
        source_name = (
            "model.language_model." + name.removeprefix("model.")
            if name.startswith("model.")
            else name
        )
        weights.append((source_name, torch.randn_like(parameter)))
    weights.append(("model.visual.patch_embed.proj.weight", torch.ones(1)))

    loaded = model.load_weights(iter(weights))

    assert loaded == {name for name, _ in model.named_parameters()}


def test_gdn_keeps_checkpoint_fp32_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BF16 model construction must not narrow recurrent or gated-norm state."""
    with set_default_torch_dtype(torch.bfloat16):
        model = _build_model(monkeypatch)

    linear_attention = model.model.layers[0].linear_attn
    assert linear_attention.A_log.dtype == torch.float32
    assert linear_attention.norm.weight.dtype == torch.float32
    assert linear_attention.dt_bias.dtype == torch.bfloat16
    assert linear_attention.in_proj_qkv.weight.dtype == torch.bfloat16


def test_validate_adapter_rejects_tensor_parallelism() -> None:
    """The Transformers-owned adapter must fail closed outside TP1."""
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _validate_adapter_config(_vllm_config(tensor_parallel_size=2))


def test_mamba_state_dtype_uses_checkpoint_value() -> None:
    """The custom architecture should preserve Qwen3.5's FP32 recurrent state."""
    config = _vllm_config()

    state_dtypes = HyperQwen3_5ForCausalLM.get_mamba_state_dtype_from_config(config)

    assert state_dtypes == (torch.bfloat16, torch.float32)


def test_text_only_mrope_positions() -> None:
    """Text requests should use equal temporal, height, and width positions."""
    model = HyperQwen3_5ForCausalLM.__new__(HyperQwen3_5ForCausalLM)

    positions, delta = model.get_mrope_input_positions([4, 5, 6], [])

    assert positions.tolist() == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    assert delta == 0
    with pytest.raises(ValueError, match="text-only"):
        model.get_mrope_input_positions([4], [object()])
