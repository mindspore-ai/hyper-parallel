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
"""CPU unit tests for Native and Hyper Qwen3 vLLM adapter contracts."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring,unnecessary-lambda
# pylint: disable=unexpected-keyword-arg,not-callable

import sys
from types import SimpleNamespace
from types import MethodType
from typing import Any

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from vllm.config.compilation import CompilationMode

import rl.roles.rollout.vllm_plugin as plugin_module
import rl.roles.rollout.vllm_qwen3 as adapter_module
import rl.roles.rollout.vllm_qwen3_common as common_module
from rl.roles.model import ModelRegistration
from rl.roles.rollout.vllm import VLLMGenerationEngine


def _model() -> ModelRegistration:
    """Return the supported tied Qwen3 identity."""
    return ModelRegistration(
        "qwen",
        "qwen3",
        "/model",
        "/tokenizer",
        "Qwen3ForCausalLM",
        "qwen3",
        "qwen3",
        True,
    )


def test_native_and_hyper_qwen3_select_expected_vllm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native keeps upstream architecture while Hyper registers one adapter idempotently."""
    native = VLLMGenerationEngine(
        _model(),
        {"vllm": {"model_implementation": "native"}},
        client=object(),
    )
    hyper = VLLMGenerationEngine(
        _model(),
        {"vllm": {"model_implementation": "hyper"}},
        client=object(),
    )
    native_command = native._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access
    hyper_command = hyper._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access
    registered: dict[str, str] = {}

    class FakeRegistry:
        """Expose the two vLLM registry methods used by the plugin."""

        @staticmethod
        def get_supported_archs() -> tuple[str, ...]:
            return tuple(registered)

        @staticmethod
        def register_model(architecture: str, model_class: str) -> None:
            registered[architecture] = model_class

    versions = {"vllm": "0.22.1", "vllm-ascend": "0.22.1rc1"}
    hook_modes: list[bool] = []
    monkeypatch.setattr(plugin_module, "package_version", versions.__getitem__)
    monkeypatch.setattr(
        plugin_module,
        "install_vllm_weight_sync_hooks",
        lambda *, private_lifecycle: hook_modes.append(private_lifecycle),
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=FakeRegistry))

    plugin_module.register_hyper_models()
    plugin_module.register_hyper_models()
    diagnostics: list[str] = []
    profiles: list[str] = []
    monkeypatch.setenv("HYPER_RL_TEST_QWEN3_RMS_NORM", "1")
    monkeypatch.setenv("HYPER_RL_CONSISTENCY_PROFILE", "profile")
    monkeypatch.setattr(
        plugin_module,
        "install_qwen3_rollout_rms_norm_diagnostic",
        lambda: diagnostics.append("rms"),
    )
    monkeypatch.setattr(
        plugin_module,
        "install_rollout_consistency_profile",
        lambda profile: profiles.append(profile),
    )
    plugin_module.register_hyper_models()

    assert "--hf-overrides" not in native_command
    override = hyper_command[hyper_command.index("--hf-overrides") + 1]
    assert override == '{"architectures": ["HyperQwen3ForCausalLM"]}'
    assert registered["HyperQwen3ForCausalLM"] == (
        "rl.roles.rollout.vllm_qwen3:HyperQwen3ForCausalLM"
    )
    assert hook_modes == [False, True, False, True, False, True]
    assert diagnostics == ["rms"]
    assert profiles == ["profile"]


def test_hyper_qwen3_adapter_preserves_forward_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter applies QKV, RoPE, paged attention, and output projection in order."""
    events: list[str] = []

    class RecordingProjection(torch.nn.Module):
        """Return a fixed-width view while recording projection order."""

        def __init__(self, name: str, width: int) -> None:
            super().__init__()
            self.name = name
            self.width = width

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            events.append(self.name)
            repeats = self.width // hidden_states.shape[-1] + 1
            return hidden_states.repeat(1, 1, repeats)[..., : self.width]

    class RecordingNorm(torch.nn.Module):
        """Record normalization without changing values."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            events.append(self.name)
            return values

    class RecordingAttention(torch.nn.Module):
        """Record the packed QKV shapes consumed by paged attention."""

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            events.append("paged-attention")
            assert query.shape == (3, 4)
            assert key.shape == (3, 2)
            assert value.shape == (3, 2)
            return query

    wrapper = object.__new__(common_module.Qwen3PagedAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper.q_proj = RecordingProjection("q-proj", 4)
    wrapper.k_proj = RecordingProjection("k-proj", 2)
    wrapper.v_proj = RecordingProjection("v-proj", 2)
    wrapper.q_norm = RecordingNorm("q-norm")
    wrapper.k_norm = RecordingNorm("k-norm")
    wrapper.o_proj = RecordingProjection("o-proj", 4)
    wrapper.attention = RecordingAttention()
    wrapper.num_heads = 2
    wrapper.num_key_value_heads = 1
    wrapper.head_dim = 2

    def rotary(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cos, sin
        events.append("rope")
        return query, key

    monkeypatch.setattr(common_module, "apply_rotary_pos_emb", rotary)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    position_embeddings = (torch.ones((1, 3, 2)), torch.zeros((1, 3, 2)))

    output, cache = wrapper(hidden_states, position_embeddings, None)

    assert output.shape == hidden_states.shape
    assert cache is None
    assert events == [
        "q-proj",
        "k-proj",
        "v-proj",
        "q-norm",
        "k-norm",
        "rope",
        "paged-attention",
        "o-proj",
    ]


def test_hyper_qwen3_adapter_loads_supported_weight_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP1 and TP2 loaders preserve mapped Qwen3 weight and tied-name contracts."""
    default_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def default_loader(parameter: torch.Tensor, loaded: torch.Tensor) -> None:
        default_calls.append((parameter, loaded))
        with torch.no_grad():
            parameter.copy_(loaded)

    class FakeDistributedTensor:
        """Expose the TP-local tensor selected by the mocked placement."""

        @staticmethod
        def to_local() -> torch.Tensor:
            return torch.tensor([3.0, 4.0])

    distributed_calls: list[tuple[Any, tuple[Any, ...]]] = []

    def distribute(
        loaded: torch.Tensor,
        mesh: Any,
        placements: tuple[Any, ...],
        *,
        src_data_rank: Any,
    ) -> FakeDistributedTensor:
        del loaded, src_data_rank
        distributed_calls.append((mesh, placements))
        return FakeDistributedTensor()

    monkeypatch.setattr(adapter_module, "default_weight_loader", default_loader)
    monkeypatch.setattr(adapter_module, "distribute_tensor", distribute)
    tp1_parameter = torch.nn.Parameter(torch.zeros(2))
    tp2_parameter = torch.nn.Parameter(torch.zeros(2))

    adapter_module._load_parameter(tp1_parameter, torch.tensor([1.0, 2.0]))  # pylint: disable=protected-access
    adapter_module._load_parameter(  # pylint: disable=protected-access
        tp2_parameter,
        torch.arange(4, dtype=torch.float32),
        tp_mesh="tp-mesh",
        placements=("shard-0",),
    )
    torch.testing.assert_close(tp1_parameter, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(tp2_parameter, torch.tensor([3.0, 4.0]))
    assert len(default_calls) == 1
    assert distributed_calls == [("tp-mesh", ("shard-0",))]
    assert adapter_module._map_weight_name("model.rotary_emb.inv_freq") is None  # pylint: disable=protected-access
    assert adapter_module._map_weight_name(  # pylint: disable=protected-access
        "model.layers.0.weight"
    ) == "model.layers.0.weight"


def _vllm_config(tp_size: int = 1) -> Any:
    """Build the supported adapter configuration around a tiny Qwen3 model."""
    hf_config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    hf_config.is_causal = True
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        quant_config=None,
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


def test_qwen3_adapter_builds_tp_mesh_from_vllm_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production mesh adapter retains the exact vLLM process group and ranks."""
    process_group = object()
    tp_group = SimpleNamespace(device_group=process_group, ranks=(2, 3))
    marked: list[Any] = []

    class Mesh:
        rank_list = (2, 3)

        @staticmethod
        def get_group() -> Any:
            return process_group

    monkeypatch.setattr(adapter_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(adapter_module, "mark_created_groups", lambda group: marked.append(group))
    monkeypatch.setattr(
        adapter_module.DeviceMesh,
        "from_group",
        lambda group, **kwargs: Mesh()
        if group is process_group and kwargs == {"device_type": "npu", "mesh_dim_names": ("tp",)}
        else None,
    )

    mesh = adapter_module._device_mesh_from_vllm_tp()  # pylint: disable=protected-access

    assert isinstance(mesh, Mesh)
    assert marked == [process_group]


def test_qwen3_adapter_initializes_tiny_supported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production adapter initialization replaces every tiny Qwen3 attention layer."""
    replacements: list[str] = []

    class AttentionLeaf(torch.nn.Module):
        def __init__(
            self,
            _attention: Any,
            *,
            vllm_config: Any,
            prefix: str,
            family: str,
        ) -> None:
            super().__init__()
            assert vllm_config.parallel_config.tensor_parallel_size == 1
            assert family == "HyperQwen3ForCausalLM"
            replacements.append(prefix)

    monkeypatch.setattr(adapter_module, "Qwen3PagedAttention", AttentionLeaf)

    model = adapter_module.HyperQwen3ForCausalLM(vllm_config=_vllm_config(), prefix="ut")

    assert replacements == ["ut.model.layers.0.self_attn"]
    assert isinstance(model.model.layers[0].self_attn, AttentionLeaf)
    assert model._tp_mesh is None
    assert model._tp_placements == {}
    assert model.get_input_embeddings() is model.model.embed_tokens
    embedded = model.get_input_embeddings(torch.tensor([1, 2]))
    assert embedded.shape == (2, 8)
    assert model.compute_logits(torch.zeros((2, 8))).shape == (2, 16)


def test_qwen3_adapter_loads_complete_tp_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production loader ignores rotary metadata and fills a tied head from embedding."""
    model = object.__new__(adapter_module.HyperQwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(4, 2)
    model.lm_head = torch.nn.Linear(2, 4, bias=False)
    model.config = SimpleNamespace(tie_word_embeddings=True)
    model._tp_mesh = None
    model._tp_placements = {}

    def named_parameters(self: Any, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        return iter(
            (
                ("model.embed_tokens.weight", model.model.embed_tokens.weight),
                ("lm_head.weight", model.lm_head.weight),
            )
        )

    model.named_parameters = MethodType(named_parameters, model)
    loaded_embedding = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    monkeypatch.setattr(
        adapter_module,
        "default_weight_loader",
        lambda parameter, loaded: parameter.data.copy_(loaded),
    )

    loaded = model.load_weights(
        [
            ("model.rotary_emb.inv_freq", torch.ones(2)),
            ("model.embed_tokens.weight", loaded_embedding),
        ]
    )

    assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
    torch.testing.assert_close(model.model.embed_tokens.weight, loaded_embedding)
    torch.testing.assert_close(model.lm_head.weight, loaded_embedding)


def test_qwen3_attention_leaf_initializes_vllm_paged_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter retains every projection and configures the TP-local attention leaf."""
    captured = {}

    class PagedAttention(torch.nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            captured.update(kwargs)

    projection = torch.nn.Linear(8, 8, bias=False)
    attention = SimpleNamespace(
        q_proj=projection,
        k_proj=projection,
        v_proj=projection,
        o_proj=projection,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        head_dim=4,
        config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=2),
        scaling=0.5,
        layer_idx=0,
        sliding_window=None,
    )
    monkeypatch.setattr(common_module, "Attention", PagedAttention)

    leaf = common_module.Qwen3PagedAttention(
        attention,
        vllm_config=_vllm_config(tp_size=2),
        prefix="model.layers.0.self_attn",
        family="HyperQwen3ForCausalLM",
    )

    assert leaf.q_proj is projection
    assert leaf.num_heads == 4
    assert leaf.num_key_value_heads == 2
    assert captured == {
        "num_heads": 2,
        "head_size": 4,
        "scale": 0.5,
        "num_kv_heads": 1,
        "cache_config": None,
        "quant_config": None,
        "per_layer_sliding_window": None,
        "prefix": "model.layers.0.self_attn.attn",
    }


def test_qwen3_adapter_applies_tp_sharding_plan_and_tied_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP initialization records planner metadata for both tied checkpoint names."""
    calls = []

    class AttentionLeaf(torch.nn.Module):
        def __init__(self, _attention: Any, **_kwargs: Any) -> None:
            super().__init__()

    class Planner:
        @staticmethod
        def plan(model: Any, mesh: Any, **kwargs: Any) -> str:
            calls.append(("plan", model, mesh, kwargs))
            return "plan"

    monkeypatch.setattr(adapter_module, "Qwen3PagedAttention", AttentionLeaf)
    monkeypatch.setattr(
        adapter_module,
        "validate_model_compatibility",
        lambda model, **kwargs: calls.append(("validate", model, kwargs)),
    )
    monkeypatch.setattr(adapter_module, "_device_mesh_from_vllm_tp", lambda: "mesh")
    monkeypatch.setattr(adapter_module, "ShardingPlanner", Planner)
    monkeypatch.setattr(
        adapter_module,
        "apply_sharding_plan",
        lambda model, plan, mesh, **kwargs: (
            model,
            {"model.embed_tokens.weight": (("shard-vocab",), "source-mesh")},
        ),
    )

    model = adapter_module.HyperQwen3ForCausalLM(
        vllm_config=_vllm_config(tp_size=2), prefix="ut"
    )

    assert model._tp_mesh == "mesh"
    assert model._tp_placements == {
        "model.embed_tokens.weight": ("shard-vocab",),
        "lm_head.weight": ("shard-vocab",),
    }
    assert calls[0][0] == "validate"
    assert calls[0][2] == {"tp_size": 2}
    assert calls[1][0] == "plan"
    assert calls[1][2] == "mesh"


def test_qwen3_adapter_forward_accepts_ids_and_precomputed_embeddings() -> None:
    """Packed forward normalizes both position forms and applies layers plus final norm."""

    class Layer:
        def __call__(self, hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            assert kwargs["attention_mask"] is None
            assert kwargs["past_key_values"] is None
            assert kwargs["use_cache"] is False
            return hidden + 1

    model = object.__new__(adapter_module.HyperQwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(
        embed_tokens=lambda ids: torch.nn.functional.one_hot(ids, num_classes=4).float(),
        rotary_emb=lambda hidden, positions: (hidden, positions),
        layers=[Layer(), Layer()],
        norm=lambda hidden: hidden * 2,
    )
    input_ids = torch.tensor([1, 2, 3])

    from_ids = model.forward(input_ids, positions=torch.tensor([0, 1, 2]))
    embeddings = torch.ones((3, 4))
    from_embeddings = model.forward(
        None,
        positions=torch.tensor([[0, 1, 2]]),
        inputs_embeds=embeddings,
    )

    torch.testing.assert_close(
        from_ids,
        (torch.nn.functional.one_hot(input_ids, num_classes=4).float() + 2) * 2,
    )
    torch.testing.assert_close(from_embeddings, torch.full((3, 4), 6.0))
