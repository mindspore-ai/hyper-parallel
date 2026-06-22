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

"""Shard ops cases for ``mint.nn.functional.one_hot``."""
import numpy as np
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

_ONEHOT_1D = np.random.RandomState(1).randint(0, 32, size=(64,)).astype(np.int64)
_ONEHOT_2D = np.random.RandomState(2).randint(0, 64, size=(16, 32)).astype(np.int64)
_ONEHOT_3D = np.random.RandomState(4).randint(0, 32, size=(8, 16, 12)).astype(np.int64)
_AUTO_1D = np.random.RandomState(5).randint(0, 8, size=(128,)).astype(np.int64)
_AUTO_1D_2 = np.random.RandomState(6).randint(0, 15, size=(128,)).astype(np.int64)


def _one_hot_32(x):
    return ms.mint.nn.functional.one_hot(x, 32)


def _one_hot_64(x):
    return ms.mint.nn.functional.one_hot(x, 64)


def _one_hot_neg1(x):
    return ms.mint.nn.functional.one_hot(x, -1)


register(OpShardCase(
    name="one_hot_ext_ops_data_parallel_1d",
    fn=_one_hot_32,
    inputs=[InputSpec(shape=(64,), data=_ONEHOT_1D, dtype="int64")],
    placements=[(Shard(0),)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="one_hot_ext_ops_data_parallel_2d",
    fn=_one_hot_64,
    inputs=[InputSpec(shape=(16, 32), data=_ONEHOT_2D, dtype="int64")],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="one_hot_ext_ops_replicate_all",
    fn=_one_hot_32,
    inputs=[InputSpec(shape=(64,), data=_ONEHOT_1D, dtype="int64")],
    placements=[(Replicate(),)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="one_hot_ext_ops_3d_data_parallel",
    fn=_one_hot_32,
    inputs=[InputSpec(shape=(8, 16, 12), data=_ONEHOT_3D, dtype="int64")],
    placements=[(Shard(0), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="one_hot_ext_ops_auto_depth_skewed",
    fn=_one_hot_neg1,
    inputs=[InputSpec(shape=(128,), data=_AUTO_1D, dtype="int64")],
    placements=[(Shard(0),)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="one_hot_ext_ops_auto_depth_uniform",
    fn=_one_hot_neg1,
    inputs=[InputSpec(shape=(128,), data=_AUTO_1D_2, dtype="int64")],
    placements=[(Shard(0),)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
