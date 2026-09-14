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
"""Tests for the TP-extended MoE shared-expert template."""

from types import SimpleNamespace

from torch import nn

from hyper_parallel.distributed._builder.shared_expert_template import (
    build_shared_expert_specs,
    matches_shared_expert_template,
)
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class _SharedExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(8, 16)
        self.linear_fc2 = nn.Linear(16, 8)


class _MoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8)])
        self.token_dispatcher = object()
        self.shared_expert = _SharedExpert()


class _Model(nn.Module):
    def __init__(self, moe: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList([moe])


def test_shared_expert_linears_are_replicated_and_preserve_sequence_shards():
    """Shared-expert params stay replicated while their sequence shard is preserved."""
    model = _Model(_MoE())

    specs = build_shared_expert_specs(model)

    assert matches_shared_expert_template(model)
    assert set(specs) == {
        "layers.0.shared_expert.linear_fc1",
        "layers.0.shared_expert.linear_fc2",
    }
    for spec in specs.values():
        assert all(isinstance(value["tp"], Replicate) for value in spec.params.values())
        assert spec.in_src["hidden_states"]["tp"] == Shard(1)
        assert spec.in_dst["hidden_states"]["tp"] == Shard(1)
        assert spec.out_src["output"]["tp"] == Shard(1)
        assert spec.out_dst["output"]["tp"] == Shard(1)


def test_shared_expert_template_rejects_non_dispatcher_mlp():
    """An MLP without a shared expert is not claimed by the template."""
    model = _Model(_SharedExpert())

    assert not matches_shared_expert_template(model)
    assert not build_shared_expert_specs(model)


def test_planner_structural_provider_installs_shared_expert_specs():
    """The planner's structural provider installs specs for shared-expert linears."""
    model = _Model(_MoE())
    plan = SimpleNamespace(modules={})

    ShardingPlanner._derive_structural_template_specs(plan, model)  # pylint: disable=protected-access

    assert "layers.0.shared_expert.linear_fc1" in plan.modules
    assert "layers.0.shared_expert.linear_fc2" in plan.modules
