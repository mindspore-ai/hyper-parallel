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

from torch import nn  # pylint: disable=forbidden-backend-import

import hyper_models._transformers.infrastructure as infrastructure_module
from hyper_models.components.distributed.config import FSDP2Config
from hyper_models.components.distributed.fsdp2 import FSDP2Manager
from hyper_models.components.distributed.infrastructure import MeshContext
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_parallelize_receives_tp_fqns_after_compile(monkeypatch) -> None:
    """
    Feature: Dual-mode compile and FSDP integration order.
    Description: Pass FQN-keyed TP metadata through compilation into FSDP parallelize.
    Expectation: Compilation runs first and FSDP receives the unchanged metadata mapping.
    """
    manager = FSDP2Manager(FSDP2Config(), MeshContext())
    model = nn.Linear(4, 4)
    tp_grad_info_by_fqn = {"weight": object()}
    call_order = []

    def _compile_model(input_model: nn.Module, **kwargs: object) -> nn.Module:
        """Record model compilation."""
        assert input_model is model
        assert not kwargs
        call_order.append("compile")
        return input_model

    def _parallelize(
        input_model: nn.Module,
        tp_grad_info: dict | None,
    ) -> nn.Module:
        """Record FSDP wrapping."""
        assert input_model is model
        assert tp_grad_info is tp_grad_info_by_fqn
        call_order.append("parallelize")
        return input_model

    manager.parallelize = Mock(side_effect=_parallelize)
    monkeypatch.setattr(infrastructure_module.torch, "compile", _compile_model)

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
    assert call_order == ["compile", "parallelize"]
