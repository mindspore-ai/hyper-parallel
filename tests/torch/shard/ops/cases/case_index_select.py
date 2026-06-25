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
"""Shard ops cases for ``torch.index_select``."""
import numpy as np
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

np.random.seed(42)
_INPUT_2D_NP = np.random.randn(8, 4).astype(np.float32)
_INPUT_3D_NP = np.random.randn(4, 6, 8).astype(np.float32)
_IDX_13_NP = np.array([1, 3], dtype=np.int64)
_IDX_12_NP = np.array([1, 2], dtype=np.int64)
_IDX_2_NP = np.array([2], dtype=np.int64)
_IDX_DUP_NP = np.array([1, 1, 3, 2, 1], dtype=np.int64)
_IDX_OOO_NP = np.array([6, 1, 7, 0, 3], dtype=np.int64)

_INPUT_2D_SPEC = InputSpec(shape=(8, 4), data=_INPUT_2D_NP, dtype="float32")
_INPUT_3D_SPEC = InputSpec(shape=(4, 6, 8), data=_INPUT_3D_NP, dtype="float32")
_IDX_13_SPEC = InputSpec(shape=(2,), data=_IDX_13_NP, dtype="int64")
_IDX_12_SPEC = InputSpec(shape=(2,), data=_IDX_12_NP, dtype="int64")
_IDX_2_SPEC = InputSpec(shape=(1,), data=_IDX_2_NP, dtype="int64")
_IDX_DUP_SPEC = InputSpec(shape=(5,), data=_IDX_DUP_NP, dtype="int64")
_IDX_OOO_SPEC = InputSpec(shape=(5,), data=_IDX_OOO_NP, dtype="int64")

# Replicated placements for 1D index tensors on 2D mesh
_IDX_REP = (Replicate(), Replicate())


def _index_select_dim0(x, index):
    return torch.index_select(x, 0, index)


def _index_select_dim1(x, index):
    return torch.index_select(x, 1, index)


def _index_select_dim2(x, index):
    return torch.index_select(x, 2, index)


def _index_select_neg1(x, index):
    return torch.index_select(x, -1, index)


register(OpShardCase(
    name="index_select_ops_basic",
    fn=_index_select_dim1,
    inputs=[_INPUT_2D_SPEC, _IDX_13_SPEC],
    placements=[(Shard(0), Replicate()), _IDX_REP],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_select_ops_3d",
    fn=_index_select_dim1,
    inputs=[_INPUT_3D_SPEC, _IDX_13_SPEC],
    placements=[(Shard(0), Replicate(), Shard(1)), _IDX_REP],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_select_ops_neg_dim",
    fn=_index_select_neg1,
    inputs=[_INPUT_2D_SPEC, _IDX_13_SPEC],
    placements=[(Shard(0), Replicate()), _IDX_REP],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_select_ops_2d_dim0",
    fn=_index_select_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_13_SPEC],
    placements=[(Replicate(), Shard(1)), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_select_ops_2d_dim1",
    fn=_index_select_dim1,
    inputs=[_INPUT_2D_SPEC, _IDX_13_SPEC],
    placements=[(Shard(0), Replicate()), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_select_ops_3d_dim1",
    fn=_index_select_dim1,
    inputs=[_INPUT_3D_SPEC, _IDX_13_SPEC],
    placements=[(Shard(0), Replicate(), Shard(1)), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="index_select_ops_single_elem",
    fn=_index_select_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_2_SPEC],
    placements=[(Replicate(), Shard(1)), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="index_select_ops_sharded_dim0",
    fn=_index_select_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_13_SPEC],
    placements=[(Shard(0), Replicate()), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_select_ops_sharded_dim1",
    fn=_index_select_dim1,
    inputs=[_INPUT_2D_SPEC, _IDX_12_SPEC],
    placements=[(Replicate(), Shard(1)), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="index_select_ops_sharded_dim2",
    fn=_index_select_dim2,
    inputs=[_INPUT_3D_SPEC, _IDX_13_SPEC],
    placements=[(Shard(0), Replicate(), Shard(1)), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="index_select_ops_duplicate",
    fn=_index_select_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_DUP_SPEC],
    placements=[(Shard(0), Replicate()), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="index_select_ops_out_of_order",
    fn=_index_select_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_OOO_SPEC],
    placements=[(Shard(0), Replicate()), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="index_select_ops_replicated",
    fn=_index_select_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_13_SPEC],
    placements=[(Replicate(), Replicate()), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_select_ops_neg_sharded",
    fn=_index_select_neg1,
    inputs=[_INPUT_2D_SPEC, _IDX_12_SPEC],
    placements=[(Replicate(), Shard(1)), _IDX_REP],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
