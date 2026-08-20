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
"""Tests for activation checkpointing in model infrastructure."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from hyper_models._transformers.infrastructure import (
    _apply_activation_checkpointing,
    apply_model_infrastructure,
)
from hyper_models.components.distributed.activation_checkpointing import (
    _find_transformer_layer_container_infos,
    make_selective_checkpoint_context_fn,
)
from hyper_models.components.distributed.fsdp2 import FSDP2Manager
from hyper_models.config.resolver import resolve_component
from hyper_models.trainer.config import ActivationCheckpointConfig


class _CountingLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.forward_calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(hidden_states)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _CountingLinear()
        self.forward_calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.mlp(hidden_states)


class _Backbone(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(num_layers))


class _Model(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.model = _Backbone(num_layers)
        self.config = SimpleNamespace(use_cache=True)


def test_activation_checkpoint_config_accepts_off_full_or_selective() -> None:
    disabled = resolve_component(
        yaml.safe_load("mode: off"),
        expected_type=ActivationCheckpointConfig,
        path="$.activation_checkpoint",
    )
    null_disabled = resolve_component(
        {"mode": None},
        expected_type=ActivationCheckpointConfig,
        path="$.activation_checkpoint",
    )
    enabled = resolve_component(
        {"mode": "full"},
        expected_type=ActivationCheckpointConfig,
        path="$.activation_checkpoint",
    )

    assert disabled.mode == "off"
    assert null_disabled.mode is None
    assert enabled.mode == "full"
    selective = resolve_component(
        {"mode": "selective"},
        expected_type=ActivationCheckpointConfig,
        path="$.activation_checkpoint",
    )
    assert selective.mode == "selective"


@pytest.mark.parametrize("mode", [None, "off"])
def test_disabled_activation_checkpoint_does_not_apply(mode: str | None) -> None:
    model = _Model()
    original_layers = list(model.model.layers)

    with patch(
        "hyper_models._transformers.infrastructure._apply_activation_checkpointing"
    ) as apply_checkpoint:
        result = apply_model_infrastructure(model, activation_checkpoint=mode)

    assert result is model
    apply_checkpoint.assert_not_called()
    assert list(model.model.layers) == original_layers
    assert model.config.use_cache is True


def test_full_activation_checkpoint_wraps_layer_submodules() -> None:
    model = _Model(num_layers=3)
    original_layers = list(model.model.layers)
    original_mlps = [layer.mlp for layer in original_layers]

    result = _apply_activation_checkpointing(model, "full")

    assert result is model
    assert model.config.use_cache is False
    assert list(model.model.layers) == original_layers
    for layer, original_mlp in zip(model.model.layers, original_mlps):
        assert layer.mlp._wrapped_module is original_mlp


def test_full_activation_checkpoint_recomputes_submodules_during_backward() -> None:
    model = _Model(num_layers=2)
    original_layers = list(model.model.layers)
    original_mlps = [layer.mlp for layer in original_layers]
    _apply_activation_checkpointing(model, "full")

    hidden_states = torch.randn(2, 4, requires_grad=True)
    for layer in model.model.layers:
        hidden_states = layer(hidden_states)
    hidden_states.sum().backward()

    assert [layer.forward_calls for layer in original_layers] == [1, 1]
    assert [mlp.forward_calls for mlp in original_mlps] == [2, 2]


def test_selective_activation_checkpoint_wraps_layers_with_context_fn() -> None:
    model = _Model(num_layers=2)
    original_layers = list(model.model.layers)

    _apply_activation_checkpointing(model, "selective")

    assert model.config.use_cache is False
    for wrapped, original in zip(model.model.layers, original_layers):
        assert wrapped._wrapped_module is original
        assert callable(wrapped.checkpoint_kwargs["context_fn"])

    hidden_states = torch.randn(2, 4, requires_grad=True)
    for layer in model.model.layers:
        hidden_states = layer(hidden_states)
    hidden_states.sum().backward()

    assert [layer.forward_calls for layer in original_layers] == [2, 2]


def test_selective_checkpoint_recomputes_mutated_topk_output() -> None:
    def router_loss(hidden_states: torch.Tensor) -> torch.Tensor:
        router_probs = torch.softmax(hidden_states, dim=-1)
        routing_weights, selected_experts = torch.topk(router_probs, 2, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        selected_states = torch.gather(hidden_states, -1, selected_experts)
        return (routing_weights * selected_states).sum()

    baseline_input = torch.randn(3, 8, requires_grad=True)
    checkpoint_input = baseline_input.detach().clone().requires_grad_(True)

    baseline_loss = router_loss(baseline_input)
    baseline_loss.backward()
    checkpoint_loss = checkpoint(
        router_loss,
        checkpoint_input,
        use_reentrant=False,
        context_fn=make_selective_checkpoint_context_fn(),
    )
    checkpoint_loss.backward()

    torch.testing.assert_close(checkpoint_loss, baseline_loss)
    torch.testing.assert_close(checkpoint_input.grad, baseline_input.grad)


def test_selective_context_registers_profiler_and_fsdp_runtime_ops() -> None:
    with (
        patch(
            "hyper_models.components.distributed.activation_checkpointing."
            "ensure_profiler_ops_sac_ignored"
        ) as ensure_profiler_ops,
        patch(
            "hyper_models.components.distributed.activation_checkpointing."
            "ensure_fsdp_ops_sac_ignored"
        ) as ensure_fsdp_ops,
    ):
        context_fn = make_selective_checkpoint_context_fn()

    ensure_profiler_ops.assert_called_once_with()
    ensure_fsdp_ops.assert_called_once_with()
    assert callable(context_fn)


def test_selective_checkpointing_falls_back_for_kv_sharing(caplog) -> None:
    class _KVLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)
            self.input_layernorm = nn.LayerNorm(4)
            self.post_attention_layernorm = nn.LayerNorm(4)

    model = _Model(num_layers=0)
    model.model.layers = nn.ModuleList([_KVLayer()])
    model.config = SimpleNamespace(
        use_cache=True,
        text_config=SimpleNamespace(num_kv_shared_layers=1, use_cache=True),
    )
    layer = model.model.layers[0]
    original_attention = layer.self_attn
    wrapped_submodules = []

    with (
        caplog.at_level("WARNING"),
        patch(
            "hyper_models.components.distributed.activation_checkpointing.checkpoint_wrapper",
            side_effect=lambda submodule: wrapped_submodules.append(submodule)
            or submodule,
        ),
    ):
        _apply_activation_checkpointing(model, "selective")

    assert model.model.layers[0] is layer
    assert layer.self_attn is original_attention
    assert wrapped_submodules == [
        layer.mlp,
        layer.input_layernorm,
        layer.post_attention_layernorm,
    ]
    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True
    assert "falling back to submodule activation checkpointing" in caplog.text


def test_layer_container_discovery_finds_all_known_vlm_towers_once() -> None:
    class Qwen2VLForConditionalGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([_Layer(), _Layer()])
            self.model.visual = nn.Module()
            self.model.visual.blocks = nn.ModuleList([_Layer()])

            # Deprecated transformers aliases point to the same modules. The
            # canonical candidates must win without double-counting containers.
            self.language_model = self.model.language_model
            self.visual = self.model.visual

    model = Qwen2VLForConditionalGeneration()

    groups = _find_transformer_layer_container_infos(model)
    assert {group_name: len(containers[0][2]) for group_name, containers in groups.items()} == {
        "language": 2,
        "vision": 1,
    }
    assert groups["language"][0][2] is model.model.language_model.layers
    assert groups["vision"][0][2] is model.model.visual.blocks


def test_known_model_with_stale_paths_does_not_use_heuristic(caplog) -> None:
    class Qwen2VLForConditionalGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unrelated_modules = nn.ModuleList([_Layer(), _Layer()])

    model = Qwen2VLForConditionalGeneration()

    with caplog.at_level("WARNING"):
        containers = _find_transformer_layer_container_infos(model)

    assert containers == {}
    assert "resolved no modules" in caplog.text


def test_unknown_model_uses_numeric_module_dict_heuristic() -> None:
    class UnknownModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapters = nn.ModuleDict({"default": _Layer()})
            self.blocks = nn.ModuleDict({"0": _Layer(), "2": _Layer()})

    model = UnknownModel()
    original_layers = list(model.blocks.values())

    groups = _find_transformer_layer_container_infos(model)
    assert list(groups) == ["unknown"]
    assert len(groups["unknown"]) == 1
    assert list(groups["unknown"][0][2].values()) == original_layers

    _apply_activation_checkpointing(model, "full")

    assert list(model.blocks.values()) == original_layers
    for layer in original_layers:
        assert hasattr(layer.mlp, "_wrapped_module")


class _HFGradientCheckpointingLayer(GradientCheckpointingLayer):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(4, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class _HFGradientCheckpointingModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_HFGradientCheckpointingLayer()])
        self.config = SimpleNamespace(use_cache=True)
        self.native_gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.native_gradient_checkpointing_kwargs = kwargs


def test_full_checkpointing_uses_hf_native_api_for_eligible_language_model() -> None:
    model = _HFGradientCheckpointingModel()

    with patch(
        "hyper_models.components.distributed.activation_checkpointing.checkpoint_wrapper"
    ) as wrapper:
        _apply_activation_checkpointing(model, "full")

    assert model.native_gradient_checkpointing_kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": True}
    }
    wrapper.assert_not_called()


def test_compile_disables_hf_native_checkpointing() -> None:
    model = _HFGradientCheckpointingModel()
    wrapper_calls = []
    original_mlp = model.model.layers[0].mlp

    with patch(
        "hyper_models.components.distributed.activation_checkpointing.checkpoint_wrapper",
        side_effect=lambda layer, **kwargs: wrapper_calls.append((layer, kwargs)) or layer,
    ):
        _apply_activation_checkpointing(model, "full", enable_compile=True)

    assert model.native_gradient_checkpointing_kwargs is None
    assert wrapper_calls == [(original_mlp, {})]


def test_full_submodule_checkpointing_skips_attention_for_kv_sharing() -> None:
    class _KVLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)
            self.input_layernorm = nn.LayerNorm(4)
            self.post_attention_layernorm = nn.LayerNorm(4)

    model = _Model(num_layers=0)
    model.model.layers = nn.ModuleList([_KVLayer()])
    model.config = SimpleNamespace(
        use_cache=True,
        text_config=SimpleNamespace(num_kv_shared_layers=1, use_cache=True),
    )
    layer = model.model.layers[0]
    original_attention = layer.self_attn
    wrapped_submodules = []

    with patch(
        "hyper_models.components.distributed.activation_checkpointing.checkpoint_wrapper",
        side_effect=lambda submodule: wrapped_submodules.append(submodule) or submodule,
    ):
        _apply_activation_checkpointing(model, "full")

    assert layer.self_attn is original_attention
    assert wrapped_submodules == [
        layer.mlp,
        layer.input_layernorm,
        layer.post_attention_layernorm,
    ]
    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True


def test_activation_checkpoint_is_after_sharding_and_before_fsdp() -> None:
    model = _Model()
    events = []
    plan = SimpleNamespace(modules={"model.layers.0": object()})
    planner = SimpleNamespace(
        plan=lambda *args, **kwargs: (events.append("sharding_plan") or plan)
    )
    mesh = SimpleNamespace(
        device_mesh=object(),
        tp_size=2,
        cp_size=1,
        ep_size=1,
        sequence_parallel=False,
        loss_parallel=False,
    )
    fsdp_manager = object.__new__(FSDP2Manager)

    def _parallelize(self, wrapped_model, **kwargs):
        del self, kwargs
        events.append("fsdp")
        return wrapped_model

    fsdp_manager.parallelize = MethodType(_parallelize, fsdp_manager)

    def _apply_sharding(sharded_model, *args, **kwargs):
        del args, kwargs
        events.append("sharding_apply")
        return sharded_model, {}

    def _checkpoint_wrapper(layer):
        events.append("checkpoint")
        return layer

    with (
        patch(
            "hyper_models.components.distributed.sharding_applier."
            "apply_sharding_plan",
            side_effect=_apply_sharding,
        ),
        patch(
            "hyper_models.components.distributed.activation_checkpointing.checkpoint_wrapper",
            side_effect=_checkpoint_wrapper,
        ),
    ):
        apply_model_infrastructure(
            model,
            mesh=mesh,
            sharding_planner=planner,
            fsdp2_manager=fsdp_manager,
            activation_checkpoint="full",
        )

    assert events == [
        "sharding_plan",
        "sharding_apply",
        "checkpoint",
        "checkpoint",
        "fsdp",
    ]
