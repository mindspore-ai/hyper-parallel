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
"""Shard ops cases for ``DTensor.copy_`` / ``zero_`` / ``fill_``."""
import numpy as np

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _copy(dst, src):
    dst.copy_(src)
    return dst


def _zero(x):
    x.zero_()
    return x


def _fill(x, val):
    x.fill_(val)
    return x


register(OpShardCase(
    name="dtensor_copy_ops_same_placement",
    fn=_copy,
    inputs=[
        InputSpec(shape=(8, 4), init="zeros", seed=42),
        InputSpec(shape=(8, 4), init="randn", seed=43),
    ],
    placements=[(Shard(0),), (Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-6),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="dtensor_copy_ops_dtype_cast",
    fn=_copy,
    inputs=[
        InputSpec(shape=(8, 4), init="zeros", seed=42),
        InputSpec(shape=(8, 4), init="randn", seed=43, dtype="float16"),
    ],
    placements=[(Shard(0),), (Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="dtensor_copy_ops_scalar_broadcast",
    fn=_copy,
    inputs=[
        InputSpec(shape=(8, 4), init="zeros", seed=42),
        # 0-d tensor src broadcast via copy_, matching old test
        # ``dst.copy_(torch.tensor(7.0))``; data must be a numpy array.
        InputSpec(shape=(), data=np.array(7.0, dtype=np.float32), dtype="float32"),
    ],
    placements=[(Shard(0),), (Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-6),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="dtensor_zero_ops",
    fn=_zero,
    inputs=[InputSpec(shape=(8, 4), init="randn", seed=42)],
    placements=[(Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-6, atol=1e-7),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="dtensor_fill_ops",
    fn=_fill,
    inputs=[InputSpec(shape=(8, 4), init="zeros", seed=42)],
    extra_inputs=[4.25],
    placements=[(Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-6, atol=1e-7),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))
