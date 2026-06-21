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

"""Shard ops cases for ``scatter_`` (inplace scatter value)."""
import numpy as np

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

_IDX1 = np.random.RandomState(1).randint(0, 256, size=(16, 8)).astype(np.int64)
_IDX2 = np.random.RandomState(2).randint(0, 16, size=(8, 256)).astype(np.int64)


def _scatter_value_dim1(x, index):
    return x.scatter_(1, index, 10.0)


def _scatter_value_dim0(x, index):
    return x.scatter_(0, index, 5.0)


register(OpShardCase(
    name="inplace_scatter_value_ops_data_parallel",
    fn=_scatter_value_dim1,
    inputs=[
        InputSpec(shape=(16, 256), init="randn", seed=42),
        InputSpec(shape=(16, 8), data=_IDX1, dtype="int64"),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="inplace_scatter_value_ops_model_parallel",
    fn=_scatter_value_dim0,
    inputs=[
        InputSpec(shape=(16, 256), init="randn", seed=42),
        InputSpec(shape=(8, 256), data=_IDX2, dtype="int64"),
    ],
    placements=[(Replicate(), Shard(1)), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
