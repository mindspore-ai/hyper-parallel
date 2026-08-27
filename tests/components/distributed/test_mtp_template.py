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
"""Tests for the structurally matched MTP TP template."""

from types import SimpleNamespace

from torch import nn

from hyper_parallel.auto_models.components.distributed.mtp_template import (
    build_mtp_specs,
    matches_mtp_template,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import (
    ShardingPlanner,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class _TinyMtpLayer(nn.Module):
    def __init__(self) -> None:
        """Build the complete structural contract of one MTP layer."""
        super().__init__()
        self.mtp_block = nn.Identity()
        self.prev_norm = nn.LayerNorm(8)
        self.emb_norm = nn.LayerNorm(8)
        self.prev_proj = nn.Linear(16, 8, bias=False)


class _LookalikeLayer(nn.Module):
    def __init__(self) -> None:
        """Build a projection with the same name but no MTP structure."""
        super().__init__()
        self.prev_proj = nn.Linear(16, 8, bias=False)


class _TinyModel(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        """Place a candidate layer at the canonical decoder-layer path."""
        super().__init__()
        self.config = SimpleNamespace(architectures=["UnrelatedArchitecture"])
        self.layers = nn.ModuleList([layer])


class _FakeTpMesh:
    mesh_dim_names = ("tp",)
    mesh_shape = (2,)


def test_mtp_template_declares_replicated_prev_proj_with_sequence_shards():
    """MTP prev_proj keeps its weight replicated across TP ranks."""
    model = _TinyModel(_TinyMtpLayer())

    specs = build_mtp_specs(model)

    assert matches_mtp_template(model)
    assert set(specs) == {"layers.0.prev_proj"}
    spec = specs["layers.0.prev_proj"]
    assert isinstance(spec.params["weight"]["tp"], Replicate)
    assert spec.in_src["input"]["tp"] == Shard(1)
    assert spec.in_dst["input"]["tp"] == Shard(1)
    assert spec.out_src["output"]["tp"] == Shard(1)
    assert spec.out_dst["output"]["tp"] == Shard(1)


def test_mtp_template_rejects_a_prev_proj_name_without_mtp_structure():
    model = _TinyModel(_LookalikeLayer())

    assert not matches_mtp_template(model)
    assert not build_mtp_specs(model)


def test_planner_covers_all_mtp_specific_trainable_parameters():
    """The plan declares sharding for every MTP-specific trainable parameter."""
    model = _TinyModel(_TinyMtpLayer())

    plan = ShardingPlanner().plan(model, _FakeTpMesh(), tp_size=2)

    declared = {
        f"{fqn}.{param_name}"
        for fqn, spec in plan.modules.items()
        for param_name in (spec.params or {})
    }
    assert set(dict(model.named_parameters())) == declared
    assert isinstance(plan.modules["layers.0.prev_proj"].params["weight"]["tp"], Replicate)
