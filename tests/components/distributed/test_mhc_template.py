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
"""Tests for the MHC tensor-parallel parameter template."""

from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.auto_models.components.distributed.mhc_template import (
    build_mhc_specs,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import (
    ShardingPlanner,
)
from hyper_parallel.core.dtensor.placement_types import Replicate


class _TinyMhc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch_alpha = nn.Parameter(torch.ones(4))
        self.branch_beta = nn.Parameter(torch.ones(4))
        self.norm_gamma = nn.Parameter(torch.ones(16))
        self.phi = nn.Linear(16, 4, bias=False)


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param_sink_k_pe = nn.Parameter(torch.ones(8, 4))
        self.param_sink_compressed_kv = nn.Parameter(torch.ones(8, 4))


class _TinyLayer(nn.Module):
    def __init__(self, *, merge: bool) -> None:
        super().__init__()
        self.self_attention = _TinyAttention()
        self.attn_mhc_pre_module = _TinyMhc()
        self.mlp_mhc_pre_module = _TinyMhc()
        if merge:
            self.merge_mhc_module = _TinyMhc()


class _TinyVlModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Deliberately unrelated: DSA/MHC templates must match structure, not
        # config.architectures or model_type.
        self.config = SimpleNamespace(
            architectures=["UnrelatedArchitecture"],
        )
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([
            _TinyLayer(merge=False),
            _TinyLayer(merge=True),
        ])


class _FakeTpMesh:
    mesh_dim_names = ("tp",)
    mesh_shape = (2,)


def test_mhc_template_owns_only_mhc_parameters():
    model = _TinyVlModel()

    specs = build_mhc_specs(model)

    assert specs
    assert all("mhc" in fqn for fqn in specs)
    for spec in specs.values():
        assert spec.is_boundary is False
        assert all(
            isinstance(placements["tp"], Replicate)
            for placements in spec.params.values()
        )


def test_planner_composes_dsa_and_mhc_templates_by_structure():
    """Independent structural templates jointly cover their parameters."""
    model = _TinyVlModel()

    plan = ShardingPlanner().plan(model, _FakeTpMesh(), tp_size=2)

    declared = {}
    for fqn, spec in plan.modules.items():
        for param_name, placements in (spec.params or {}).items():
            declared[f"{fqn}.{param_name}"] = placements
            assert spec.is_boundary is False
            assert isinstance(placements["tp"], Replicate)

    assert set(declared) == set(dict(model.named_parameters()))
