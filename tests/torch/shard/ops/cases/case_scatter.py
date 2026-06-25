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
"""Shard ops cases for ``Tensor.scatter``."""
import numpy as np

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

np.random.seed(1234)
_SCATTER_INDEX_NP = np.random.randint(0, 8, (8, 8)).astype(np.int64)

np.random.seed(2345)
_SCATTER_INDEX_SCALAR_NP = np.random.randint(0, 8, (8, 4)).astype(np.int64)


def _scatter_basic(x, index, src):
    return x.scatter(1, index, src)


def _scatter_scalar_src(x, index):
    return x.scatter(1, index, 3.14159)


register(OpShardCase(
    name="scatter_ops_basic",
    fn=_scatter_basic,
    inputs=[
        InputSpec(shape=(8, 8), init="zeros"),
        InputSpec(shape=(8, 8), data=_SCATTER_INDEX_NP, dtype="int64"),
        InputSpec(shape=(8, 8), init="randn", seed=1234),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="scatter_ops_scalar_src",
    fn=_scatter_scalar_src,
    inputs=[
        InputSpec(shape=(8, 8), init="zeros"),
        InputSpec(shape=(8, 4), data=_SCATTER_INDEX_SCALAR_NP, dtype="int64"),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
