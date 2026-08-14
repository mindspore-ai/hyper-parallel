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
"""Contract tests for the Hyper Qwen3.5 vLLM adapter."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

pytest.importorskip("vllm")

from vllm import ModelRegistry
from vllm.config.compilation import CUDAGraphMode, CompilationMode

import rl.roles.rollout.vllm_qwen3_5 as qwen3_5_adapter
from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.qwen3_5.model import Qwen3_5Attention
from hyper_parallel.models.qwen3_5.parallelize import (
    _TP_PROFILE_INFERENCE_REPLICATED,
    _build_qwen3_5_tp_plans,
)
from rl.roles.rollout.vllm_plugin import (
    HYPER_QWEN3_5_ARCHITECTURE,
    register_hyper_models,
)
from rl.roles.rollout.vllm_qwen3_5 import (
    HyperQwen3_5ForCausalLM,
    _alignment_enabled,
    _build_hyper_config,
    _configure_attention_alignment,
    _enable_gdn_alignment,
    _map_weight_name,
    _reset_rope_inv_freq_from_cpu,
    _validate_adapter_config,
    _validate_loaded_weight_shards,
)


@pytest.fixture(autouse=True)
def _disable_alignment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unrelated adapter tests independent of the caller environment."""
    monkeypatch.delenv("HYPER_VLLM_ALIGNMENT", raising=False)


def _vllm_config(**overrides):
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=8,
        rope_parameters={
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.75,
            "mrope_section": [1, 1, 1],
        },
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        full_attention_interval=4,
        attn_output_gate=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        mamba_ssm_dtype="float32",
    )
    hf_config = SimpleNamespace(
        tie_word_embeddings=True,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=text_config,
            hf_config=hf_config,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        quant_config=None,
        speculative_config=None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_registry_inspects_hybrid_text_model() -> None:
    """vLLM should discover generation and hybrid-state interfaces."""
    register_hyper_models()
    model_info = ModelRegistry.models[HYPER_QWEN3_5_ARCHITECTURE].inspect_model_cls()

    assert model_info.is_text_generation_model
    assert model_info.has_inner_state
    assert model_info.is_hybrid
    assert not model_info.supports_multimodal
    assert not model_info.supports_pp


def test_build_hyper_config_uses_checkpoint_geometry() -> None:
    """The adapter should preserve Qwen3.5 geometry and RoPE fields."""
    config = _build_hyper_config(_vllm_config())

    assert config.hidden_size == 16
    assert config.layer_types[-1] == "full_attention"
    assert config.rope_theta == 1_000_000.0
    assert config.partial_rotary_factor == 0.75
    assert config.mrope_section == [1, 1, 1]
    assert config.tie_word_embeddings


@pytest.mark.parametrize(
    "source_name,target_name,shard_id",
    [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            None,
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.layers.0.linear_attn.in_proj_z.weight",
            None,
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            None,
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
            None,
        ),
        (
            "model.language_model.layers.3.mlp.up_proj.weight",
            "model.layers.3.mlp.up_proj.weight",
            None,
        ),
    ],
)
def test_map_weight_name(
    source_name: str,
    target_name: str,
    shard_id: int | tuple[int, ...] | None,
) -> None:
    """Checkpoint parameters should retain native Hyper module names."""
    assert _map_weight_name(source_name) == (target_name, shard_id)


def test_validate_adapter_accepts_compatible_tensor_parallelism() -> None:
    """The adapter should accept TP sizes that divide every sharded dimension."""
    config = _vllm_config()
    config.model_config.hf_text_config.num_key_value_heads = 2
    config.parallel_config.tensor_parallel_size = 2

    _validate_adapter_config(config)


def test_validate_adapter_rejects_incompatible_tensor_parallelism() -> None:
    """The adapter should reject TP sizes that cannot partition Qwen3.5 heads."""
    config = _vllm_config()
    config.parallel_config.tensor_parallel_size = 3

    with pytest.raises(ValueError, match="must be divisible by TP size 3"):
        _validate_adapter_config(config)


def test_validate_adapter_rejects_incompatible_kv_head_topology() -> None:
    """Native Hyper TP does not silently replicate KV heads."""
    config = _vllm_config()
    config.model_config.hf_text_config.num_attention_heads = 6
    config.model_config.hf_text_config.num_key_value_heads = 3
    config.model_config.hf_text_config.intermediate_size = 30
    config.model_config.hf_text_config.linear_num_key_heads = 6
    config.model_config.hf_text_config.linear_num_value_heads = 6
    config.parallel_config.tensor_parallel_size = 2

    with pytest.raises(ValueError, match="num_key_value_heads=3 must be divisible"):
        _validate_adapter_config(config)


def test_inference_tp_plan_keeps_boundaries_replicated() -> None:
    """Inference TP should shard features without Sequence Parallelism."""
    plans = _build_qwen3_5_tp_plans(_TP_PROFILE_INFERENCE_REPLICATED)

    assert plans.root["model.embed_tokens"].output_layouts[0].is_replicate()
    assert plans.root["lm_head"].output_layouts[0].is_replicate()
    assert plans.full_attention["self_attn.q_proj"].output_layouts[0].is_shard()
    assert plans.full_attention["self_attn.o_proj"].output_layouts[0].is_replicate()
    assert plans.linear_attention["linear_attn.out_proj"].output_layouts[0].is_replicate()


def test_tp_mesh_reuses_vllm_device_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hyper mesh must retain vLLM's exact device process group."""
    process_group = object()
    tp_group = SimpleNamespace(device_group=process_group, ranks=[2, 3])
    mesh = SimpleNamespace(
        rank_list=(2, 3),
        get_group=lambda: process_group,
    )
    marked_groups = []
    mesh_calls = []

    def fake_from_group(group: object, **kwargs: object) -> object:
        mesh_calls.append((group, kwargs))
        return mesh

    monkeypatch.setattr(qwen3_5_adapter, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(qwen3_5_adapter, "mark_created_groups", marked_groups.append)
    monkeypatch.setattr(
        qwen3_5_adapter.DeviceMesh,
        "from_group",
        fake_from_group,
    )

    result = qwen3_5_adapter._device_mesh_from_vllm_tp()

    assert result is mesh
    assert marked_groups == [process_group]
    assert mesh_calls[0][0] is process_group


def test_adapter_preserves_native_hyper_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM leaves must not replace Hyper embedding, Attention, MLP, or LM head."""
    class FakeAttention(nn.Module):
        """Minimal paged-attention stand-in for adapter construction."""

        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs

    class FakeGdnRuntime(nn.Module):
        """Parameter-free stand-in for vLLM's request-state runtime."""

        def __init__(self, config: object, vllm_config: object, prefix: str, **kwargs: object) -> None:
            super().__init__()
            del config, vllm_config, kwargs
            self.prefix = prefix
            self.tp_size = 1

    monkeypatch.setattr(qwen3_5_adapter, "Attention", FakeAttention)
    monkeypatch.setattr(qwen3_5_adapter, "QwenGatedDeltaNetAttention", FakeGdnRuntime)

    model = HyperQwen3_5ForCausalLM(vllm_config=_vllm_config())
    full_attention_layer = model.model.layers[-1]
    linear_attention_layer = model.model.layers[0]

    assert isinstance(model.model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert isinstance(full_attention_layer.self_attn, Qwen3_5Attention)
    assert isinstance(full_attention_layer.mlp, SwiGLUMLP)
    assert isinstance(full_attention_layer.self_attn.q_proj, nn.Linear)
    assert isinstance(linear_attention_layer.mlp, SwiGLUMLP)
    assert isinstance(linear_attention_layer.linear_attn.in_proj_qkv, nn.Linear)
    assert not list(linear_attention_layer.linear_attn.state_runtime.parameters())
    linear_attention_layer.linear_attn.to_empty(device="meta")
    assert (
        linear_attention_layer.linear_attn.state_runtime.dt_bias
        is linear_attention_layer.linear_attn.dt_bias
    )
    assert (
        linear_attention_layer.linear_attn.state_runtime.A_log
        is linear_attention_layer.linear_attn.A_log
    )


def test_validate_phase_one_uses_checkpoint_ssm_dtype() -> None:
    """The unique Hyper architecture should retain Qwen3.5's canonical FP32 state."""
    config = _vllm_config()

    _validate_adapter_config(config)

    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


def test_alignment_rejects_speculative_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")
    config = _vllm_config(speculative_config=SimpleNamespace())

    with pytest.raises(ValueError, match="does not support speculative decoding"):
        _validate_adapter_config(config)


def test_alignment_rejects_prefix_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")
    config = _vllm_config()
    config.cache_config.enable_prefix_caching = True
    config.cache_config.mamba_cache_mode = "align"

    with pytest.raises(ValueError, match="does not support prefix caching"):
        _validate_adapter_config(config)


def test_alignment_rejects_chunked_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")
    config = _vllm_config()
    config.scheduler_config.enable_chunked_prefill = True

    with pytest.raises(ValueError, match="does not support chunked prefill"):
        _validate_adapter_config(config)


def test_alignment_rejects_graph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")
    config = _vllm_config()
    config.compilation_config.cudagraph_mode = object()

    with pytest.raises(ValueError, match="does not support graph capture"):
        _validate_adapter_config(config)


def test_alignment_gdn_selects_confirmed_compatible_ops() -> None:
    gdn = SimpleNamespace(
        hyper_qwen3_5_alignment_api=1,
        supported_causal_conv_activation_modes={"fused", "separate_bf16"},
        supported_gdn_gating_modes={"fused", "torch"},
        supported_gdn_recurrence_modes={"fla", "torch"},
        gdn_gating_mode="fused",
        gdn_recurrence_mode="fla",
    )

    _enable_gdn_alignment(gdn)

    assert gdn.causal_conv_activation_mode == "separate_bf16"
    assert gdn.gdn_gating_mode == "torch"
    assert gdn.gdn_recurrence_mode == "torch"


def test_alignment_gdn_requires_qwen3_5_patch_marker() -> None:
    with pytest.raises(ValueError, match="Qwen3.5 causal-conv"):
        _enable_gdn_alignment(SimpleNamespace())


def test_alignment_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPER_VLLM_ALIGNMENT", raising=False)

    assert _alignment_enabled() is False


def test_alignment_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "unknown")

    with pytest.raises(ValueError, match="must be true or false"):
        _alignment_enabled()


def test_alignment_attention_selects_fusion_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")
    implementation = SimpleNamespace(
        hyper_qwen3_5_alignment_api=1,
        supported_prefill_attention_modes={"fia", "fusion"},
        prefill_attention_mode="fia",
    )

    _configure_attention_alignment(SimpleNamespace(impl=implementation))

    assert implementation.prefill_attention_mode == "fusion"


def test_disabled_alignment_uses_backend_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPER_VLLM_ALIGNMENT", raising=False)
    implementation = SimpleNamespace(prefill_attention_mode="fia")

    _configure_attention_alignment(SimpleNamespace(impl=implementation))

    assert implementation.prefill_attention_mode == "fia"


def test_alignment_attention_requires_qwen3_5_patch_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYPER_VLLM_ALIGNMENT", "true")

    with pytest.raises(ValueError, match="Qwen3.5 fusion-prefill"):
        _configure_attention_alignment(SimpleNamespace(impl=SimpleNamespace()))


def test_reset_rope_inv_freq_uses_cpu_reference() -> None:
    rope = SimpleNamespace(dim=4, theta=10_000.0, inv_freq=torch.zeros(2))

    _reset_rope_inv_freq_from_cpu(rope, torch.bfloat16)

    expected = 1.0 / (rope.theta ** (torch.arange(0, rope.dim, 2).float() / rope.dim))
    assert torch.equal(rope.inv_freq, expected.to(torch.bfloat16).float())


def test_mamba_state_dtype_uses_checkpoint_before_model_init() -> None:
    """Hybrid cache sizing should see the checkpoint FP32 state before construction."""
    config = _vllm_config()

    state_dtypes = HyperQwen3_5ForCausalLM.get_mamba_state_dtype_from_config(config)

    assert state_dtypes == (torch.bfloat16, torch.float32)


def test_validate_loaded_weight_shards_requires_every_fused_input() -> None:
    """Strict loading should reject a missing native GDN parameter."""
    parameter_name = "model.layers.0.linear_attn.in_proj_qkv.weight"

    with pytest.raises(ValueError, match="expected shards"):
        _validate_loaded_weight_shards(
            {parameter_name},
            {},
        )

    _validate_loaded_weight_shards(
        {parameter_name},
        {parameter_name: {None}},
    )


def test_reload_synthesizes_layerwise_untied_lm_head() -> None:
    """Layerwise reload should restore a temporary lm-head copy from tied embeddings."""
    model = HyperQwen3_5ForCausalLM.__new__(HyperQwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(4, 2)
    model.lm_head = nn.Linear(2, 4, bias=False)
    model.config = SimpleNamespace(tie_word_embeddings=True)
    model._tp_load_transforms = {}
    embedding_weight = torch.arange(8, dtype=torch.float32).view(4, 2)

    loaded = model.load_weights(
        [("model.language_model.embed_tokens.weight", embedding_weight)]
    )

    assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
    assert torch.equal(model.model.embed_tokens.weight, embedding_weight)
    assert torch.equal(model.lm_head.weight, embedding_weight)
