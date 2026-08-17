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
"""Tests for FSDP metadata preparation order in model infrastructure."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

import hyper_models._transformers.infrastructure as infrastructure_module
from hyper_models.components.distributed.config import FSDP2Config
from hyper_models.components.distributed.fsdp2 import FSDP2Manager
from hyper_models.components.distributed.infrastructure import MeshContext
from tests.common.mark_utils import arg_mark


class _DeferredBufferModule(nn.Module):
    """Module with loaded state and a non-persistent derived buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.weight._is_hf_initialized = True
        self.register_buffer("inv_freq", torch.zeros(2), persistent=False)
        self._is_hf_initialized = True


class _DeferredModel(nn.Module):
    """Minimal Transformers-like model exposing guarded initialization."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _DeferredBufferModule()
        self.initialize_weights_calls = 0
        self.all_tied_weights_keys = {}
        self._keys_to_ignore_on_load_unexpected = {r"^ignored\."}

    def initialize_weights(self) -> None:
        """Initialize only tensors without the Transformers marker."""
        self.initialize_weights_calls += 1
        for module in self.modules():
            if getattr(module, "_is_hf_initialized", False):
                continue
            if isinstance(module, _DeferredBufferModule):
                if not getattr(module.weight, "_is_hf_initialized", False):
                    module.weight.data.fill_(9.0)
                if not getattr(module.inv_freq, "_is_hf_initialized", False):
                    module.inv_freq.copy_(torch.tensor([1.0, 0.5]))
            module._is_hf_initialized = True


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_parallelize_receives_tp_fqns_before_compile(monkeypatch) -> None:
    """
    Feature: Dual-mode compile and FSDP integration order.
    Description: Pass FQN-keyed TP metadata into FSDP before layer compilation.
    Expectation: FSDP receives the unchanged metadata mapping before compilation.
    """
    manager = FSDP2Manager(FSDP2Config(), MeshContext())
    model = nn.Linear(4, 4)
    tp_grad_info_by_fqn = {"weight": object()}
    call_order = []

    def _compile_model(input_model: nn.Module, config: object) -> nn.Module:
        """Record model compilation."""
        del config
        assert input_model is model
        call_order.append("compile")
        return input_model

    def _parallelize(
        input_model: nn.Module,
        tp_grad_info: dict | None,
        *,
        compile_hooks_enabled: bool = False,
    ) -> nn.Module:
        """Record FSDP wrapping."""
        assert input_model is model
        assert tp_grad_info is tp_grad_info_by_fqn
        assert compile_hooks_enabled
        call_order.append("parallelize")
        return input_model

    manager.parallelize = Mock(side_effect=_parallelize)
    monkeypatch.setattr(infrastructure_module, "apply_compile", _compile_model)

    sharding_planner = Mock()
    sharding_planner.plan.return_value = object()
    mesh = MeshContext(device_mesh=object())
    monkeypatch.setattr(
        infrastructure_module,
        "apply_sharding_plan",
        lambda input_model, plan, input_mesh, validate_mode: (
            input_model,
            tp_grad_info_by_fqn,
        ),
    )

    result = infrastructure_module.apply_model_infrastructure(
        model,
        mesh=mesh,
        sharding_planner=sharding_planner,
        fsdp2_manager=manager,
        compile_config={},
    )

    assert result is model
    assert call_order == ["parallelize", "compile"]


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_meta_pretrained_model_loads_weights_after_materialization(monkeypatch) -> None:
    """
    Feature: Deferred pretrained loading.
    Description: Materialize a meta model before invoking the Stage 3 checkpoint loader.
    Expectation: Loading receives the materialized model and requested checkpoint path.
    """
    meta_model = nn.Linear(2, 2, device="meta")
    materialized_model = nn.Linear(2, 2)
    calls = []

    def _move_model(model: nn.Module, is_meta_device: bool, device: object) -> nn.Module:
        """Record model materialization."""
        assert model is meta_model
        assert is_meta_device
        calls.append("materialize")
        return materialized_model

    checkpoint_manager = Mock()

    def _load_weights(pretrained_path: str, *, strict: bool) -> infrastructure_module.LoadReport:
        """Record deferred checkpoint loading."""
        assert pretrained_path == "checkpoint"
        assert not strict
        calls.append("load")
        return infrastructure_module.LoadReport((), (), ())

    monkeypatch.setattr(infrastructure_module, "_move_model_to_device", _move_model)
    checkpoint_manager.load_checkpoint.side_effect = _load_weights
    manager_type = Mock(return_value=checkpoint_manager)
    monkeypatch.setattr(infrastructure_module, "CheckpointManager", manager_type)

    result = infrastructure_module.apply_model_infrastructure(
        meta_model,
        is_meta_device=True,
        load_base_model=True,
        pretrained_path="checkpoint",
    )

    assert result is materialized_model
    assert calls == ["materialize", "load"]
    manager_type.assert_called_once_with(materialized_model)
    checkpoint_manager.load_checkpoint.assert_called_once_with("checkpoint", strict=False)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_non_meta_pretrained_model_is_not_loaded_twice(monkeypatch) -> None:
    """
    Feature: Existing from_pretrained behavior.
    Description: Process a non-meta model whose checkpoint was already loaded by Transformers.
    Expectation: The deferred Stage 3 loader is not called.
    """
    model = nn.Linear(2, 2)
    manager_type = Mock()
    monkeypatch.setattr(infrastructure_module, "CheckpointManager", manager_type)

    result = infrastructure_module.apply_model_infrastructure(
        model,
        is_meta_device=False,
        load_base_model=True,
        pretrained_path="checkpoint",
    )

    assert result is model
    manager_type.assert_not_called()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_initialize_model_weights_avoids_retying_parameters() -> None:
    """
    Feature: Deferred random initialization.
    Description: Prefer Transformers initialize_weights over init_weights.
    Expectation: Initialization runs without invoking the method that also ties weights.
    """
    model = nn.Linear(2, 2)
    model.initialize_weights = Mock()
    model.init_weights = Mock()

    infrastructure_module._initialize_model_weights(model)

    model.initialize_weights.assert_called_once_with()
    model.init_weights.assert_not_called()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_finalize_model_loading_initializes_derived_buffer_without_changing_loaded_weight() -> None:
    """
    Feature: Deferred pretrained finalization.
    Description: Initialize non-persistent state through the guarded model initialization contract.
    Expectation: The derived buffer is restored while the loaded parameter and its identity stay unchanged.
    """
    model = _DeferredModel()
    weight = model.block.weight
    weight_before = weight.detach().clone()
    report = infrastructure_module.LoadReport(
        loaded_keys=("block.weight",),
        missing_keys=(),
        unexpected_keys=("ignored.extra", "other.extra"),
    )

    finalized_report = infrastructure_module._finalize_model_loading(model, report, strict=True)

    assert model.block.weight is weight, (
        f"Loaded parameter identity changed: expected={id(weight)}, got={id(model.block.weight)}"
    )
    assert torch.equal(model.block.weight, weight_before), (
        f"Loaded parameter values changed: expected={weight_before}, got={model.block.weight}"
    )
    expected_inv_freq = torch.tensor([1.0, 0.5])
    assert torch.equal(model.block.inv_freq, expected_inv_freq), (
        f"Derived buffer was not initialized: expected={expected_inv_freq}, got={model.block.inv_freq}"
    )
    assert model.initialize_weights_calls == 1, (
        f"Expected one initialize_weights call, got={model.initialize_weights_calls}"
    )
    assert finalized_report.loaded_keys == ("block.weight",), (
        f"Loaded report changed unexpectedly: expected=('block.weight',), got={finalized_report.loaded_keys}"
    )
    assert finalized_report.unexpected_keys == ("other.extra",), (
        f"Unexpected-key adjustment failed: expected=('other.extra',), got={finalized_report.unexpected_keys}"
    )


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_finalize_model_loading_rejects_unresolved_missing_state_in_strict_mode() -> None:
    """
    Feature: Deferred pretrained finalization strictness.
    Description: Finalize a load report with an unresolved checkpoint-owned parameter.
    Expectation: Strict mode rejects the missing parameter before returning a model.
    """
    model = _DeferredModel()
    report = infrastructure_module.LoadReport(
        loaded_keys=(),
        missing_keys=("block.weight",),
        unexpected_keys=(),
    )

    with pytest.raises(RuntimeError, match="Checkpoint did not load 1 owned model tensors"):
        infrastructure_module._finalize_model_loading(model, report, strict=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_finalize_model_loading_rejects_meta_non_persistent_buffer() -> None:
    """
    Feature: Deferred pretrained finalization materialization checks.
    Description: Finalize a model whose non-persistent buffer remains on meta.
    Expectation: Finalization reports the invalid materialization state.
    """
    model = _DeferredModel()
    model.block.inv_freq = torch.empty(2, device="meta")
    report = infrastructure_module.LoadReport(
        loaded_keys=("block.weight",),
        missing_keys=(),
        unexpected_keys=(),
    )

    with pytest.raises(ValueError, match="tensors on meta device"):
        infrastructure_module._finalize_model_loading(model, report, strict=True)
