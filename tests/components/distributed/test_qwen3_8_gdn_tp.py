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
"""Unit tests for Qwen3.8 Gated DeltaNet tensor-parallel planning."""

import pytest
import torch
from torch import nn

from hyper_parallel.auto_models.components.distributed.packed_shard import (
    pack_tensor_for_shard,
    unpack_tensor_from_shard,
)
from hyper_parallel.auto_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    TP,
    ModuleShardingSpec,
    PackedShard,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import ShardingPlanner


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


def test_packed_shard_reorders_unequal_sections_by_rank():
    """Each rank receives its Q, K, and V slices in logical order."""
    tensor = torch.arange(16)
    placement = PackedShard(0, (4, 4, 8))

    packed = pack_tensor_for_shard(tensor, placement, world_size=2)

    torch.testing.assert_close(packed, torch.tensor([0, 1, 4, 5, 8, 9, 10, 11,
                                                     2, 3, 6, 7, 12, 13, 14, 15]))
    torch.testing.assert_close(unpack_tensor_from_shard(packed, placement, 2), tensor)


def test_packed_shard_rejects_non_divisible_section():
    """Packed section divisibility is checked independently."""
    placement = PackedShard(0, (3, 3, 6))

    with pytest.raises(ValueError, match="not divisible by TP size"):
        pack_tensor_for_shard(torch.arange(12), placement, world_size=2)


def test_gdn_parameter_classifier_covers_tp_parameters():
    """All GDN trainable parameters receive an explicit semantic role."""
    roles = ParameterClassifier().classify(FakeModel())

    assert roles["linear_attn.in_proj_qkv.weight"] == ParamRole.GDN_PACKED_QKV
    assert roles["linear_attn.conv1d.weight"] == ParamRole.GDN_PACKED_QKV
    assert roles["linear_attn.in_proj_z.weight"] == ParamRole.COLWISE
    assert roles["linear_attn.in_proj_b.weight"] == ParamRole.COLWISE
    assert roles["linear_attn.in_proj_a.weight"] == ParamRole.COLWISE
    assert roles["linear_attn.out_proj.weight"] == ParamRole.ROWWISE
    assert roles["linear_attn.A_log"] == ParamRole.GDN_HEAD
    assert roles["linear_attn.dt_bias"] == ParamRole.GDN_HEAD


def test_gdn_tp_spec_uses_same_packed_sections_for_projection_and_conv():
    """Planner binds Q/K/V sections and TP-local cached dimensions together."""
    module = FakeGatedDeltaNet()
    spec = ModuleShardingSpec(params={
        "in_proj_qkv.weight": {TP: PackedShard(0, (8, 8))},
        "conv1d.weight": {TP: PackedShard(0, (8, 8))},
    })

    ShardingPlanner._configure_gdn_tp_spec(  # pylint: disable=protected-access
        spec,
        module,
        "model.layers.0.linear_attn",
    )

    expected = PackedShard(0, (4, 4, 8))
    assert spec.params["in_proj_qkv.weight"][TP] == expected
    assert spec.params["conv1d.weight"][TP] == expected
    assert spec.tp_divide_attrs == [
        "num_v_heads", "num_k_heads", "key_dim", "value_dim", "conv_dim",
    ]
