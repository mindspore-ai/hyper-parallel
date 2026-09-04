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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.auto_models._transformers import checkpoint_loader
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
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard


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


@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_loader_builds_expected_packed_shard(monkeypatch, rank):
    """FSDP and TP loading preserves logical sections in a packed weight."""
    fsdp_size = 2
    tp_size = 2
    placement = PackedShard(0, (4, 4, 8))
    full_weight = torch.arange(16 * 3).reshape(16, 3)
    mesh = SimpleNamespace(size=lambda mesh_dim: (fsdp_size, tp_size)[mesh_dim])
    layout = SimpleNamespace(
        # A real FSDP+TP Layout exposes tensor-dimension aliases here. They
        # preserve nested shard order but intentionally contain no PackedShard.
        alias_placements=(("tp", "fsdp_shard"), "None"),
        placements=(StridedShard(0, split_factor=tp_size), placement),
        mesh=mesh,
    )
    target = torch.empty(4, 3)
    target._sharding_spec = layout  # pylint: disable=protected-access

    class LocalDTensor:
        """Minimal distribute_tensor result exposing the rank-local tensor."""

        def __init__(self, local_tensor: torch.Tensor) -> None:
            """Store one simulated rank-local tensor."""
            self._local_tensor = local_tensor

        def to_local(self) -> torch.Tensor:
            """Return the simulated local tensor."""
            return self._local_tensor

    def fake_distribute_tensor(
        tensor: torch.Tensor,
        target_mesh: object,
        placements: object,
        src_data_rank: int | None = None,
    ) -> LocalDTensor:
        """Simulate TP-first then FSDP slicing encoded by alias placements."""
        assert target_mesh is mesh
        assert placements == layout.alias_placements
        assert src_data_rank is None
        tp_local = tensor.chunk(tp_size, dim=0)[rank]
        return LocalDTensor(tp_local.chunk(fsdp_size, dim=0)[0])

    monkeypatch.setattr(checkpoint_loader, "distribute_tensor", fake_distribute_tensor)

    local_weight = checkpoint_loader._shard_for_target(  # pylint: disable=protected-access
        "model.layers.0.mlp.experts.gate_up_proj",
        full_weight,
        target,
    )
    q_weight, k_weight, v_weight = full_weight.split(placement.sections, dim=0)
    expected_tp_local = torch.cat(
        [section.chunk(tp_size, dim=0)[rank] for section in (q_weight, k_weight, v_weight)],
        dim=0,
    )
    expected = expected_tp_local.chunk(fsdp_size, dim=0)[0]

    torch.testing.assert_close(local_weight, expected)


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
