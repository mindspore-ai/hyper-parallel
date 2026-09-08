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
"""Shard ops cases for ``npu_mhc_post``."""
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.custom_ops.experimental import npu_mhc_post
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _npu_mhc_post(x, h_res, h_out, h_post):
    return npu_mhc_post(x, h_res, h_out, h_post)


# ----- BSND replicated (2-device) -----

register(OpShardCase(
    name="mhc_post_ops_bsnd_replicated",
    fn=_npu_mhc_post,
    inputs=[
        InputSpec(shape=(2, 4, 4, 1280), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(2, 4, 4, 4), init="randn", dtype="float32", seed=43),
        InputSpec(shape=(2, 4, 1280), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(2, 4, 4), init="randn", dtype="float32", seed=45),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level0",),
))

# ----- BSND DP (2-device) -----

register(OpShardCase(
    name="mhc_post_ops_bsnd_dp",
    fn=_npu_mhc_post,
    inputs=[
        InputSpec(shape=(2, 4, 4, 1280), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(2, 4, 4, 4), init="randn", dtype="float32", seed=43),
        InputSpec(shape=(2, 4, 1280), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(2, 4, 4), init="randn", dtype="float32", seed=45),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level0",),
))

# ----- BSND DP+CP+TP (8-device) -----

register(OpShardCase(
    name="mhc_post_ops_bsnd_dp_cp_tp",
    fn=_npu_mhc_post,
    inputs=[
        InputSpec(shape=(2, 4, 4, 1280), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(2, 4, 4, 4), init="randn", dtype="float32", seed=43),
        InputSpec(shape=(2, 4, 1280), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(2, 4, 4), init="randn", dtype="float32", seed=45),
    ],
    placements=[
        (Shard(0), Shard(1), Shard(1)),
        (Shard(0), Shard(1), Shard(1)),
        (Shard(0), Shard(1), Shard(1)),
        (Shard(0), Shard(1), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level0",),
))

# ----- TND DP+CP/TP (4-device) -----

register(OpShardCase(
    name="mhc_post_ops_tnd_dp_cp_tp",
    fn=_npu_mhc_post,
    inputs=[
        InputSpec(shape=(8, 4, 1280), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(8, 4, 4), init="randn", dtype="float32", seed=43),
        InputSpec(shape=(8, 1280), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(8, 4), init="randn", dtype="float32", seed=45),
    ],
    placements=[
        (Shard(0), Shard(0)),
        (Shard(0), Shard(0)),
        (Shard(0), Shard(0)),
        (Shard(0), Shard(0)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))
