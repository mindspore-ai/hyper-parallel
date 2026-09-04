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
"""Unit tests for Gated DeltaNet tensor-parallel planning."""

from __future__ import annotations

import torch
from torch import nn

from hyper_parallel.auto_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    TP,
    ModuleShardingSpec,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class FakeGatedDeltaNet(nn.Module):
    """Minimal Transformers-compatible GDN parameter structure."""

    def __init__(self) -> None:
        """Create unequal Q/K and V packed sections."""
        super().__init__()
        self.num_v_heads = 4
        self.num_k_heads = 2
        self.key_dim = 4
        self.value_dim = 8
        self.conv_dim = 16
        self.in_proj_qkv = nn.Linear(8, 16, bias=False)
        self.in_proj_z = nn.Linear(8, 8, bias=False)
        self.in_proj_b = nn.Linear(8, 4, bias=False)
        self.in_proj_a = nn.Linear(8, 4, bias=False)
        self.conv1d = nn.Conv1d(16, 16, 3, groups=16, bias=False)
        self.A_log = nn.Parameter(torch.ones(4))
        self.dt_bias = nn.Parameter(torch.ones(4))
        self.norm = nn.LayerNorm(2)
        self.out_proj = nn.Linear(8, 8, bias=False)


class FakeModel(nn.Module):
    """Model wrapper that provides production-like linear-attention FQNs."""

    def __init__(self) -> None:
        """Install one fake LinearAttention layer."""
        super().__init__()
        self.linear_attn = FakeGatedDeltaNet()


def test_gdn_parameter_classifier_covers_tp_parameters():
    """All GDN trainable parameters receive an explicit semantic role."""
    roles = ParameterClassifier().classify(FakeModel())

    for name in (
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    ):
        assert roles[f"linear_attn.{name}.weight"] == ParamRole.COLWISE
    for name in ("conv1d.weight", "A_log", "dt_bias"):
        assert roles[f"linear_attn.{name}"] == ParamRole.REPLICATED


def test_gdn_tp_spec_keeps_the_complete_region_replicated():
    """The stock GDN region keeps full parameters and activation dimensions."""
    module = FakeGatedDeltaNet()
    spec = ModuleShardingSpec(
        params={name: {TP: Shard(0)} for name, _ in module.named_parameters()},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={"output": {TP: Shard(1)}},
        out_dst={"output": {TP: Shard(1)}},
        tp_divide_attrs=["num_v_heads", "key_dim"],
    )

    ShardingPlanner._keep_gdn_replicated_on_tp(  # pylint: disable=protected-access
        spec,
        module,
    )

    assert all(placement[TP] == Replicate() for placement in spec.params.values())
    assert spec.in_src["hidden_states"][TP] == Shard(1)
    assert spec.in_dst["hidden_states"][TP] == Replicate()
    assert spec.out_src["output"][TP] == Replicate()
    assert spec.out_dst["output"][TP] == Shard(1)
    assert spec.tp_divide_attrs == []
