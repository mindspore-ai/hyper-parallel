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
"""Unit tests for packed gate/up tensor-parallel parameter ordering."""

import pytest
import torch

from hyper_models.components.distributed.packed_shard import (
    pack_tensor_for_shard,
    unpack_tensor_from_shard,
)
from hyper_models.components.distributed.param_role import ParamRole
from hyper_models.components.distributed.sharding_config import (
    TP,
    PackedShard,
    ShardingTemplate,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner


def test_pack_tensor_keeps_matching_gate_and_up_shards_together():
    """Each TP rank shard should contain the matching gate and up slices."""
    gate = torch.arange(8).reshape(1, 8, 1)
    up = torch.arange(100, 108).reshape(1, 8, 1)
    full = torch.cat((gate, up), dim=1)

    packed = pack_tensor_for_shard(full, PackedShard(1, parts=2), world_size=2)
    rank0, rank1 = packed.chunk(2, dim=1)

    torch.testing.assert_close(
        rank0.flatten(), torch.tensor([0, 1, 2, 3, 100, 101, 102, 103])
    )
    torch.testing.assert_close(
        rank1.flatten(), torch.tensor([4, 5, 6, 7, 104, 105, 106, 107])
    )


def test_unpack_tensor_restores_transformers_checkpoint_order():
    """Gathered rank-major shards should restore part-major checkpoint order."""
    full = torch.arange(32).reshape(2, 16, 1)
    placement = PackedShard(1, parts=2)

    packed = pack_tensor_for_shard(full, placement, world_size=4)
    restored = unpack_tensor_from_shard(packed, placement, world_size=4)

    torch.testing.assert_close(restored, full)


def test_pack_tensor_rejects_logical_projection_not_divisible_by_tp():
    """Packed validation should divide each logical part, not only the fused dim."""
    full = torch.empty(2, 12, 4)

    with pytest.raises(ValueError, match="logical projection size 6.*TP size 4"):
        pack_tensor_for_shard(full, PackedShard(1, parts=2), world_size=4)


def test_planner_uses_packed_shard_for_hf_batched_moe_gate_up():
    """HF [E,2I,H] routed experts should use the standard packed TP rule."""
    placement = ShardingPlanner._placement_for_role(  # pylint: disable=W0212
        "experts.gate_up_proj",
        ParamRole.MOE_EXPERT,
        ShardingTemplate(),
        has_tp=True,
        has_ep=False,
        ndim=3,
        param_shape=(8, 16, 32),
    )

    assert placement[TP] == PackedShard(1, parts=2)


def test_planner_uses_packed_shard_for_transposed_grouped_moe_gate_up():
    """Grouped [E,H,2I] routed experts should shard their final packed dimension."""
    placement = ShardingPlanner._placement_for_role(  # pylint: disable=W0212
        "experts.gate_and_up_projs",
        ParamRole.MOE_EXPERT,
        ShardingTemplate(),
        has_tp=True,
        has_ep=False,
        ndim=3,
        param_shape=(8, 32, 16),
    )

    assert placement[TP] == PackedShard(2, parts=2)
