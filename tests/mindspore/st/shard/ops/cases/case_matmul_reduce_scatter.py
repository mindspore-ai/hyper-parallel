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
"""Shard ops cases for ``ms.ops.matmul_reduce_scatter`` (MC2 fusion).

``matmul_reduce_scatter`` is a custom Ascend MC2 communication-computation
fusion operator (not available in ``ms.mint``).  All cases use ``# non-mint:``
and ``needs_mesh=True`` so the mesh is injected as first argument.
"""
import mindspore as ms

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _matmul_reduce_scatter(mesh, x1, x2, trans_x2=False, group_dim="tp"):
    """Wrapper for ``ms.ops.matmul_reduce_scatter`` with plain-matmul reference path."""
    # non-mint: ms.ops.matmul_reduce_scatter is a custom MC2 fusion operator.
    # Reference path (plain replicated tensors): the gathered distributed output
    # of ReduceScatter(matmul) equals the full matmul, so use plain matmul as the
    # ground truth — the generic fn(full) reference would re-run the collective
    # and mismatch. Matches the old test (ref = np.matmul(x1, x2)).
    if not isinstance(x1, DTensor):
        w = x2.swapaxes(0, 1) if trans_x2 else x2
        return ms.ops.matmul(x1, w)
    group = mesh.get_group(group_dim)
    world_size = mesh.size(mesh.mesh_dim_names.index(group_dim))
    return ms.ops.matmul_reduce_scatter(x1, x2, group, world_size, trans_x2=trans_x2)


# --- 2-card tests (level1) ---

# x1 Shard(1) on k, x2 Shard(0) on k
register(OpShardCase(
    name="mrs_ops_tp_basic",
    fn=_matmul_reduce_scatter,
    needs_mesh=True,
    inputs=[
        InputSpec(shape=(128, 512), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(512, 512), dtype="float16", init="randn", seed=43),
    ],
    # 1D mesh: placement length == mesh ndim. x1 Shard(1) on k, x2 Shard(0) on k.
    placements=[
        (Shard(1),),
        (Shard(0),),
    ],
    kwargs={"trans_x2": False, "group_dim": "tp"},
    compare=CompareSpec.allclose(rtol=2e-2, atol=1e-1),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level1",),
))

# trans_x2=True: x2 physical shape (N, K) Shard(1) on k
register(OpShardCase(
    name="mrs_ops_trans_x2",
    fn=_matmul_reduce_scatter,
    needs_mesh=True,
    inputs=[
        InputSpec(shape=(128, 512), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(512, 512), dtype="float16", init="randn", seed=43),
    ],
    # 1D mesh: x1 Shard(1) on k; x2 physical (N,K) Shard(1) on k-dim.
    placements=[
        (Shard(1),),
        (Shard(1),),
    ],
    kwargs={"trans_x2": True, "group_dim": "tp"},
    compare=CompareSpec.allclose(rtol=2e-2, atol=1e-1),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level1",),
))

# Larger M dimension (256)
register(OpShardCase(
    name="mrs_ops_large_m",
    fn=_matmul_reduce_scatter,
    needs_mesh=True,
    inputs=[
        InputSpec(shape=(256, 512), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(512, 512), dtype="float16", init="randn", seed=43),
    ],
    # 1D mesh: x1 Shard(1) on k, x2 Shard(0) on k.
    placements=[
        (Shard(1),),
        (Shard(0),),
    ],
    kwargs={"trans_x2": False, "group_dim": "tp"},
    compare=CompareSpec.allclose(rtol=2e-2, atol=1e-1),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level1",),
))

# --- 4-card test (level1 - custom MC2 fusion op) ---

# 2D dp×tp mesh, x1/x2 only sharded on tp
register(OpShardCase(
    name="mrs_ops_dp_tp",
    fn=_matmul_reduce_scatter,
    needs_mesh=True,
    inputs=[
        InputSpec(shape=(128, 512), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(512, 512), dtype="float16", init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Shard(1)),
        (Replicate(), Shard(0)),
    ],
    kwargs={"trans_x2": False, "group_dim": "tp"},
    compare=CompareSpec.allclose(rtol=2e-2, atol=1e-1),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# NOTE: no 8-card 3-D case. For matmul-family ops, sharding M (rows) and N (cols)
# is pure concatenation (no cross-shard interaction); the only non-trivial axis is
# K (contraction), whose reduction + reduce-scatter collective is already covered by
# the 2-card (mrs_ops_tp_basic) and 4-card (mrs_ops_dp_tp) cases. A 3-D mesh would
# only add the trivial M/N concat axes while triggering the CANN MC2
# HcclAllocComResourceByTiling conflict, so it adds no coverage and is omitted.
