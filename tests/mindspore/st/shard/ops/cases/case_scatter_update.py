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

"""Shard ops cases for ``scatter_update``."""
import numpy as np
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

_IDX1D = np.arange(8).astype(np.int32)
_IDX2D = np.arange(8).reshape(4, 2).astype(np.int32)


def _scatter_update(x, indices, updates):
    # non-mint: ms.mint.scatter_update not available
    return ms.ops.ScatterUpdate()(x, indices, updates)


register(OpShardCase(
    name="scatter_update_ops_data_parallel",
    fn=_scatter_update,
    inputs=[
        InputSpec(shape=(16, 256, 128), init="randn", seed=42),
        InputSpec(shape=(8,), data=_IDX1D, dtype="int32"),
        InputSpec(shape=(8, 256, 128), init="randn", seed=43),
    ],
    # scatter_update dim0 is the indexed (scatter) dim and cannot be sharded;
    # shard the non-scatter dim1 (matches old test Shard(1)). Length == mesh ndim.
    placements=[
        (Shard(1), Replicate()),
        (Replicate(), Replicate()),
        (Shard(1), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="scatter_update_ops_model_parallel",
    fn=_scatter_update,
    inputs=[
        InputSpec(shape=(16, 256, 128), init="randn", seed=42),
        InputSpec(shape=(8,), data=_IDX1D, dtype="int32"),
        InputSpec(shape=(8, 256, 128), init="randn", seed=43),
    ],
    # shard the feature dim2 (non-scatter). Length == mesh ndim.
    placements=[
        (Shard(2), Replicate()),
        (Replicate(), Replicate()),
        (Shard(2), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="scatter_update_ops_hybrid_parallel",
    fn=_scatter_update,
    inputs=[
        InputSpec(shape=(16, 256, 128), init="randn", seed=42),
        InputSpec(shape=(8,), data=_IDX1D, dtype="int32"),
        InputSpec(shape=(8, 256, 128), init="randn", seed=43),
    ],
    # shard both non-scatter dims (dim1 on dp, dim2 on tp).
    placements=[
        (Shard(1), Shard(2)),
        (Replicate(), Replicate()),
        (Shard(1), Shard(2)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="scatter_update_ops_multi_dim_indices",
    fn=_scatter_update,
    inputs=[
        InputSpec(shape=(16, 256, 128), init="randn", seed=42),
        InputSpec(shape=(4, 2), data=_IDX2D, dtype="int32"),
        InputSpec(shape=(4, 2, 256, 128), init="randn", seed=43),
    ],
    # 2D indices: x dim0 is scattered (indexed) and replicated; shard the
    # trailing feature dims of x (1,2) and the matching updates dims (2,3).
    placements=[
        (Shard(1), Shard(2)),
        (Replicate(), Replicate()),
        (Shard(2), Shard(3)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="scatter_update_ops_replicate_all",
    fn=_scatter_update,
    inputs=[
        InputSpec(shape=(16, 256, 128), init="randn", seed=42),
        InputSpec(shape=(8,), data=_IDX1D, dtype="int32"),
        InputSpec(shape=(8, 256, 128), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate(), Replicate()),
        (Replicate(),),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
