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
"""Unit tests for LlamaFactory expert-parallel preparation."""

import copy
import types

import torch
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
)

import hyper_parallel.integration.llamafactory.expert_parallel.expert_parallel as ep_prepare_mod
import hyper_parallel.integration.llamafactory.utils as llamafactory_utils
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.integration.llamafactory.expert_parallel import ep_prepare_model
from hyper_parallel.integration.llamafactory.expert_parallel.models import (
    get_expert_parallel_model_patches,
)


def _tiny_qwen3_vl_moe_config():
    """Build a small Qwen3-VL-MoE text config for CPU tests."""
    # Transformers exposes these model-specific fields through a dynamic config API.
    # pylint: disable-next=unexpected-keyword-arg
    return Qwen3VLMoeTextConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
    )


def _matching_patch_names(model):
    """Return model patch names selected from the instantiated model metadata."""
    return [
        patch.name
        for patch in get_expert_parallel_model_patches()
        if patch.supports(model)
    ]


def test_ep_prepare_model_is_exported_from_package():
    """The package-level API used by LlamaFactory trainer should be importable."""
    assert ep_prepare_model is ep_prepare_mod.ep_prepare_model


def test_ep_prepare_model_wraps_only_experts(monkeypatch):
    """EP preparation should wrap experts and leave root FSDP2 to the trainer."""
    model = torch.nn.Linear(2, 2)
    expert = torch.nn.Linear(2, 2)
    plugin = object()
    accelerator = types.SimpleNamespace(state=types.SimpleNamespace(fsdp_plugin=plugin))
    hp_args = types.SimpleNamespace(ep_size=2)
    expert_mesh = object()
    full_mesh = {("ep_replicate", "efsdp"): expert_mesh}
    ep_context = types.SimpleNamespace(
        full_mesh=full_mesh,
        expert_modules=[expert],
        ep_size=2,
    )
    fsdp_kwargs = {"mesh": expert_mesh}
    calls = []

    monkeypatch.setattr(
        ep_prepare_mod,
        "_apply_expert_parallel",
        lambda target_model, target_args: (
            calls.append(("ep", target_model, target_args)) or ep_context
        ),
    )

    def _fake_build_expert_fsdp2_kwargs(
        target_accelerator,
        target_model,
        target_args,
        target_plugin,
        target_mesh,
    ):
        calls.append(
            (
                "kwargs",
                target_accelerator,
                target_model,
                target_args,
                target_plugin,
                target_mesh,
            )
        )
        return fsdp_kwargs

    monkeypatch.setattr(
        ep_prepare_mod,
        "_build_expert_fsdp2_kwargs",
        _fake_build_expert_fsdp2_kwargs,
    )
    monkeypatch.setattr(
        ep_prepare_mod,
        "_wrap_expert_with_fsdp",
        lambda module, kwargs, ep_size: calls.append(("wrap", module, kwargs, ep_size)),
    )

    result = ep_prepare_model(model, accelerator, hp_args)

    assert result is model
    assert calls == [
        ("ep", model, hp_args),
        ("kwargs", accelerator, model, hp_args, plugin, expert_mesh),
        ("wrap", expert, fsdp_kwargs, 2),
    ]


def test_wrap_expert_with_fsdp_filters_params_and_scales_gradients(monkeypatch):
    """Expert FSDP2 should manage only expert params and scale EP gradients."""
    expert_mesh = object()
    expert = torch.nn.Linear(2, 2)
    gradient_scaling_factors = []
    expert.set_gradient_scaling_factor = gradient_scaling_factors.append
    outside_param = torch.nn.Parameter(torch.ones(1))
    captured = {}

    def _fake_fully_shard(module, **kwargs):
        captured.update(module=module, **kwargs)
        return module

    monkeypatch.setattr(ep_prepare_mod, "fully_shard", _fake_fully_shard)
    monkeypatch.setattr(
        ep_prepare_mod,
        "_collect_replicate_params",
        lambda module, shard_size: {module.bias},
    )
    monkeypatch.setattr(ep_prepare_mod, "_resolve_shard_size", lambda mesh: 2)
    # pylint: disable-next=protected-access
    ep_prepare_mod._wrap_expert_with_fsdp(
        expert,
        {
            "mesh": expert_mesh,
            "mp_policy": "mp",
            "offload_policy": "offload",
            "reshard_after_forward": False,
            "ignored_params": {expert.weight, outside_param},
        },
        ep_size=4,
    )

    assert captured == {
        "module": expert,
        "mesh": expert_mesh,
        "mp_policy": "mp",
        "offload_policy": "offload",
        "reshard_after_forward": False,
        "ignored_params": {expert.weight},
        "replicate_params": {expert.bias},
    }
    assert gradient_scaling_factors == [0.25]


def test_ep_meta_initialization_preserves_dtensor_and_hsdp_state(monkeypatch):
    """EP meta initialization should release storage without losing distributed ownership."""
    # A topology-only CPU mesh makes DTensor metadata testable without launching workers.
    monkeypatch.setattr(
        "hyper_parallel.core.dtensor.device_mesh.platform.get_rank", lambda: 0
    )
    expert_mesh = DeviceMesh(
        "cpu",
        [0],
        mesh_dim_names=("ep",),
        _init_backend=False,  # pylint: disable=protected-access
    )

    class _Expert(torch.nn.Module):
        """Small expert container carrying one already-sharded DTensor parameter."""

        def __init__(self):
            super().__init__()
            expert_weight = DTensor.from_local(
                torch.randn(4, 3),
                expert_mesh,
                (Shard(0),),
                shape=(4, 3),
                stride=(3, 1),
            )
            self.weight = torch.nn.Parameter(expert_weight, requires_grad=True)
            self.weight._layout.set_tensor_meta(  # pylint: disable=protected-access
                (4, 3), (3, 1), self.weight.dtype
            )
            self.weight._hsdp_param_initialized = True  # pylint: disable=protected-access
            self.register_buffer("scale", torch.ones(1))

    # Match fully_shard's in-place class extension without requiring an NPU device.
    expert = _Expert()
    expert.__class__ = type("HSDPTestExpert", (HSDPModule, _Expert), {})
    old_parameter = expert.weight
    old_mesh_hash = old_parameter.device_mesh.to_hash()
    old_placements = tuple(old_parameter.placements)
    old_shape = tuple(old_parameter.shape)
    old_stride = tuple(old_parameter.layout.tensor_stride)
    old_dtype = old_parameter.dtype

    class _HookMigrator:
        """Record that hooks are captured before the sharded parameter is replaced."""

        def __init__(self):
            self.saved_parameters = []

        def _save_backward_hooks(self, parameter):
            self.saved_parameters.append(parameter)

    class _HSDPParam:
        """Minimal HSDPParam reference graph used by the integration helper."""

        def __init__(self, owner, parameter):
            self.owner = owner
            self.sharded_param = parameter
            self._sharded_param_data = parameter.to_local().reshape(-1)
            self._parameter_hook_migrator = _HookMigrator()

        def _setattr_on_modules(self, parameter):
            self.owner._parameters["weight"] = parameter  # pylint: disable=protected-access

    hsdp_param = _HSDPParam(expert, old_parameter)
    hsdp_state = types.SimpleNamespace(hsdp_params=[hsdp_param])
    expert.hsdp_scheduler = types.SimpleNamespace(hsdp_state=hsdp_state)
    model = torch.nn.ModuleDict(
        {
            "experts": expert,
            "dense": torch.nn.Linear(3, 2),
        }
    )

    result = llamafactory_utils._move_unwrapped_model_state_to_meta(model)  # pylint: disable=protected-access

    assert result is model
    assert all(parameter.is_meta for parameter in model.parameters())
    assert expert.scale.is_meta
    assert isinstance(expert.weight, DTensor)
    assert expert.weight.device_mesh.to_hash() == old_mesh_hash
    assert tuple(expert.weight.placements) == old_placements
    assert tuple(expert.weight.shape) == old_shape
    assert tuple(expert.weight.layout.tensor_stride) == old_stride
    assert expert.weight.dtype == old_dtype
    assert expert.weight.requires_grad
    assert expert.weight._hsdp_param_initialized  # pylint: disable=protected-access
    assert hsdp_param.sharded_param is expert.weight
    assert hsdp_param._sharded_param_data.is_meta  # pylint: disable=protected-access
    assert hsdp_param._parameter_hook_migrator.saved_parameters == [  # pylint: disable=protected-access
        old_parameter
    ]


def test_only_qwen3_vl_moe_ep_patch_is_registered_and_selected():
    """The runtime registry should contain and select only Qwen3-VL-MoE."""
    qwen3_vl_model = Qwen3VLMoeTextSparseMoeBlock(_tiny_qwen3_vl_moe_config())

    registered_patch_names = [
        patch.name for patch in get_expert_parallel_model_patches()
    ]
    qwen3_vl_matches = _matching_patch_names(qwen3_vl_model)

    assert registered_patch_names == ["qwen3_vl_moe"], (
        "EP patch registry mismatch: "
        f"expected=['qwen3_vl_moe'], got={registered_patch_names}"
    )
    assert qwen3_vl_matches == ["qwen3_vl_moe"], (
        "Qwen3-VL-MoE patch selection mismatch: "
        f"expected=['qwen3_vl_moe'], got={qwen3_vl_matches}"
    )


def test_qwen3_vl_moe_ep_patch_matches_native_forward_and_backward():
    """The Qwen3-VL-MoE adapter should preserve native CPU numerics."""
    torch.manual_seed(7)
    native_block = Qwen3VLMoeTextSparseMoeBlock(_tiny_qwen3_vl_moe_config())
    for parameter in native_block.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
    patched_block = copy.deepcopy(native_block)

    matching_patches = [
        patch
        for patch in get_expert_parallel_model_patches()
        if patch.supports(patched_block)
    ]
    matching_patches[0].prepare(patched_block, types.SimpleNamespace(ep_size=1))

    native_input = torch.randn(2, 3, 8, requires_grad=True)
    patched_input = native_input.detach().clone().requires_grad_(True)
    native_output = native_block(native_input)
    patched_output = patched_block(patched_input)
    native_output.square().sum().backward()
    patched_output.square().sum().backward()

    torch.testing.assert_close(patched_output, native_output)
    torch.testing.assert_close(patched_input.grad, native_input.grad)
    for (native_name, native_parameter), (patched_name, patched_parameter) in zip(
        native_block.named_parameters(), patched_block.named_parameters()
    ):
        assert patched_name == native_name, (
            f"Patched parameter order mismatch: expected={native_name}, got={patched_name}"
        )
        torch.testing.assert_close(patched_parameter.grad, native_parameter.grad)
