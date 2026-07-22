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
"""Shard ops cases for ``torch_npu.npu_rotary_mul``.

.. note::
    Ascend-specific operator (``npu_*``); all cases run on NPU only
    (``npu_level1``). ``npu_rotary_mul`` applies rotary position embedding:
    ``y = x * cos + rotate(x) * sin``. Output shape equals input ``x``
    shape.

    Layout: BSND — dim 0=batch, dim 1=seq, dim 2=heads, dim 3=head_dim.

    ``npu_rotary_mul`` supports two rotary modes:
      - default (equivalent to MS mode=0, half)
      - ``rotary_mode="interleave"`` (equivalent to MS mode=1)
    MS modes 2 (quarter) and 3 (interleave-half) are not available;
    cases that use those modes in the MS ST fall back to ``interleave``
    here.
"""
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _npu_rotary_mul(x, cos, sin, **kwargs):
    # torch_npu is Ascend-only; imported inside the fn so CPU glue
    # (case discovery) does not fail on import.
    import torch_npu  # pylint: disable=C0415
    return torch_npu.npu_rotary_mul(x, cos, sin, **kwargs)


# --- 2-card tests (level1) ---

# 1-D dp mesh, all replicated, broadcast cos/sin, default mode
register(OpShardCase(
    name="npu_rotary_mul_ops_replicated",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=43),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=44),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

# 1-D dp mesh, x Shard(0), broadcast cos/sin, interleave mode
register(OpShardCase(
    name="npu_rotary_mul_ops_dp_b",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=43),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=44),
    ],
    placements=[
        (Shard(0),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"rotary_mode": "interleave"},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

# 1-D tp mesh, all Shard(2), full cos/sin, default mode
register(OpShardCase(
    name="npu_rotary_mul_ops_tp_n",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=45),
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=46),
    ],
    placements=[
        (Shard(2),),
        (Shard(2),),
        (Shard(2),),
    ],
    kwargs={"rotary_mode": "half"},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level1",),
))

# --- 4-card tests (level1) ---

# 2-D dp×tp mesh, x (Shard(0), Shard(2)), full cos/sin, default mode
register(OpShardCase(
    name="npu_rotary_mul_ops_dp_tp",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=45),
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=46),
    ],
    placements=[
        (Shard(0), Shard(2)),
        (Shard(0), Shard(2)),
        (Shard(0), Shard(2)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# 2-D dp×sp mesh, x (Shard(0), Shard(1)), broadcast cos/sin, interleave mode
register(OpShardCase(
    name="npu_rotary_mul_ops_dp_sp",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=43),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=44),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Replicate(), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    kwargs={"rotary_mode": "interleave"},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "sp"),
    tags=("npu_level1",),
))

# 2-D tp×sp mesh, all (Shard(2), Shard(1)), full cos/sin, half mode
register(OpShardCase(
    name="npu_rotary_mul_ops_tp_sp",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=45),
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=46),
    ],
    placements=[
        (Shard(1), Shard(2)),
        (Shard(1), Shard(2)),
        (Shard(1), Shard(2)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("sp", "tp"),
    tags=("npu_level1",),
))

# 2-D dp×tp mesh, x (Shard(0), Shard(2)), broadcast cos/sin, interleave mode
register(OpShardCase(
    name="npu_rotary_mul_ops_dp_tp_cos_broadcast",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=43),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=44),
    ],
    placements=[
        (Shard(0), Shard(2)),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    kwargs={"rotary_mode": "interleave"},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# --- 8-card test (level1) ---

# 3-D dp×cp×tp mesh, x sharded on B/S/N, cos/sin sharded on S,
# interleave mode
register(OpShardCase(
    name="npu_rotary_mul_ops_dp_tp_sp",
    fn=_npu_rotary_mul,
    inputs=[
        InputSpec(shape=(4, 16, 8, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=43),
        InputSpec(shape=(1, 16, 1, 32), dtype="float16", init="uniform", range=(-1.0, 1.0), seed=44),
    ],
    placements=[
        (Shard(0), Shard(1), Shard(2)),
        (Replicate(), Shard(1), Replicate()),
        (Replicate(), Shard(1), Replicate()),
    ],
    kwargs={"rotary_mode": "interleave"},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))
