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
"""Shard ops cases for ``torch.index_add`` and ``Tensor.index_add_``."""
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
_SOURCE_DIM0_NP = np.random.randn(3, 4).astype(np.float32)
_SOURCE_DIM1_NP = np.random.randn(8, 2).astype(np.float32)
_OUT_2D_NP = np.zeros((8, 4), dtype=np.float32)
_IDX_DIM0_NP = np.array([0, 2, 5], dtype=np.int64)
_IDX_DIM1_NP = np.array([1, 3], dtype=np.int64)
_IDX_DUP_NP = np.array([1, 1, 3], dtype=np.int64)

_INPUT_2D_SPEC = InputSpec(shape=(8, 4), data=_INPUT_2D_NP, dtype="float32")
_SOURCE_DIM0_SPEC = InputSpec(shape=(3, 4), data=_SOURCE_DIM0_NP, dtype="float32")
_SOURCE_DIM1_SPEC = InputSpec(shape=(8, 2), data=_SOURCE_DIM1_NP, dtype="float32")
_OUT_2D_SPEC = InputSpec(shape=(8, 4), data=_OUT_2D_NP, dtype="float32")
_IDX_DIM0_SPEC = InputSpec(shape=(3,), data=_IDX_DIM0_NP, dtype="int64")
_IDX_DIM1_SPEC = InputSpec(shape=(2,), data=_IDX_DIM1_NP, dtype="int64")
_IDX_DUP_SPEC = InputSpec(shape=(3,), data=_IDX_DUP_NP, dtype="int64")

_REP_2D = (Replicate(), Replicate())
_DP_2D = (Shard(0), Replicate())
_TP_2D = (Replicate(), Shard(1))
_IDX_REP = (Replicate(), Replicate())


def _index_add_dim0(x, index, source):
    return torch.index_add(x, 0, index, source)


def _index_add_method_dim0(x, index, source):
    return x.index_add(0, index, source)


def _index_add_dim1_alpha(x, index, source):
    return torch.index_add(x, 1, index, source, alpha=0.5)


def _index_add_neg_dim(x, index, source):
    return torch.index_add(x, -1, index, source)


def _index_add_inplace_clone(x, index, source):
    y = x.clone()
    y.index_add_(0, index, source)
    return y


def _index_add_out_dim0(x, index, source, out):
    return torch.index_add(x, 0, index, source, out=out)


register(OpShardCase(
    name="index_add_ops_replicated",
    fn=_index_add_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_DIM0_SPEC, _SOURCE_DIM0_SPEC],
    placements=[_REP_2D, _IDX_REP, _REP_2D],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_add_ops_tensor_method_dim0",
    fn=_index_add_method_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_DIM0_SPEC, _SOURCE_DIM0_SPEC],
    placements=[_TP_2D, _IDX_REP, _TP_2D],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_add_ops_dim1_alpha",
    fn=_index_add_dim1_alpha,
    inputs=[_INPUT_2D_SPEC, _IDX_DIM1_SPEC, _SOURCE_DIM1_SPEC],
    placements=[_DP_2D, _IDX_REP, _DP_2D],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_add_ops_neg_dim",
    fn=_index_add_neg_dim,
    inputs=[_INPUT_2D_SPEC, _IDX_DIM1_SPEC, _SOURCE_DIM1_SPEC],
    placements=[_DP_2D, _IDX_REP, _DP_2D],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_add_ops_duplicate_index",
    fn=_index_add_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_DUP_SPEC, _SOURCE_DIM0_SPEC],
    placements=[_TP_2D, _IDX_REP, _TP_2D],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_add_ops_inplace_clone",
    fn=_index_add_inplace_clone,
    inputs=[_INPUT_2D_SPEC, _IDX_DIM0_SPEC, _SOURCE_DIM0_SPEC],
    placements=[_TP_2D, _IDX_REP, _TP_2D],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="index_add_ops_out",
    fn=_index_add_out_dim0,
    inputs=[_INPUT_2D_SPEC, _IDX_DIM0_SPEC, _SOURCE_DIM0_SPEC, _OUT_2D_SPEC],
    placements=[_TP_2D, _IDX_REP, _TP_2D, _TP_2D],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
