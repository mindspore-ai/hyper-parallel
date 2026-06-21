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
"""Shard ops cases for ``ms.ops.rotary_position_embedding``.

``rotary_position_embedding`` is a custom Ascend operator (not available in
``ms.mint``).  All cases use ``ms.ops.rotary_position_embedding`` with a
``# non-mint:`` comment.
"""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _rpe(x, cos, sin, **kwargs):
    # non-mint: ms.ops.rotary_position_embedding is a custom Ascend operator
    return ms.ops.rotary_position_embedding(x, cos, sin, **kwargs)


# --- 2-card tests (level1) ---

# 1-D dp mesh, all replicated, broadcast cos/sin, mode=0
register(OpShardCase(
    name="rpe_ops_replicated",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Replicate(), Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
    ],
    kwargs={"mode": 0},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

# 1-D dp mesh, x Shard(0), cos/sin broadcast replicated, mode=1
register(OpShardCase(
    name="rpe_ops_dp_b",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
    ],
    kwargs={"mode": 1},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

# 1-D tp mesh, all Shard(1), full cos/sin, mode=0
register(OpShardCase(
    name="rpe_ops_tp_n",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Replicate(), Shard(1), Replicate(), Replicate()),
        (Replicate(), Shard(1), Replicate(), Replicate()),
        (Replicate(), Shard(1), Replicate(), Replicate()),
    ],
    kwargs={"mode": 0},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

# --- 4-card tests (level0) ---

# 2-D dp×tp mesh, all (Shard(0), Shard(1)), full cos/sin, mode=0
register(OpShardCase(
    name="rpe_ops_dp_tp",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate(), Replicate()),
        (Shard(0), Shard(1), Replicate(), Replicate()),
        (Shard(0), Shard(1), Replicate(), Replicate()),
    ],
    kwargs={"mode": 0},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# 2-D dp×sp mesh, x (Shard(0), Shard(2)), cos/sin broadcast replicated, mode=2
register(OpShardCase(
    name="rpe_ops_dp_sp",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Replicate(), Shard(2), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
    ],
    kwargs={"mode": 2},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# 2-D tp×sp mesh, all (Shard(1), Shard(2)), full cos/sin, mode=3
register(OpShardCase(
    name="rpe_ops_tp_sp",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Replicate(), Shard(1), Shard(2), Replicate()),
        (Replicate(), Shard(1), Shard(2), Replicate()),
        (Replicate(), Shard(1), Shard(2), Replicate()),
    ],
    kwargs={"mode": 3},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# 2-D dp×tp mesh, x (Shard(0), Shard(1)), cos/sin broadcast replicated, mode=2
register(OpShardCase(
    name="rpe_ops_dp_tp_cos_broadcast",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate(), Replicate()),
    ],
    kwargs={"mode": 2},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# --- 8-card test (level1) ---

# 3-D dp×tp×sp mesh, x (Shard(0), Shard(1), Shard(2)), cos/sin broadcast, mode=3
register(OpShardCase(
    name="rpe_ops_dp_tp_sp",
    fn=_rpe,
    inputs=[
        InputSpec(shape=(4, 4, 16, 64), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(1, 1, 16, 64), dtype="float16", init="randn", seed=44),
    ],
    # x: B->dp, N->tp, S->sp. cos/sin shard S on sp so each rank's cos/sin
    # S-slice aligns with its x S-slice (matches old test). Length == mesh ndim.
    placements=[
        (Shard(0), Shard(1), Shard(2)),
        (Replicate(), Replicate(), Shard(2)),
        (Replicate(), Replicate(), Shard(2)),
    ],
    kwargs={"mode": 3},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))
