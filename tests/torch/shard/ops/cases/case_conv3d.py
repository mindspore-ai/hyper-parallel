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
"""Shard ops cases for ``torch.nn.functional.conv3d``."""
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


_I4 = InputSpec(shape=(8, 2, 4, 4, 4), init="randn", seed=42)
_I4B4 = InputSpec(shape=(4, 2, 4, 4, 4), init="randn", seed=42)
_I4H8 = InputSpec(shape=(4, 2, 8, 4, 4), init="randn", seed=42)
_I5H4 = InputSpec(shape=(2, 4, 4, 4, 4), init="randn", seed=42)
_I4W8 = InputSpec(shape=(2, 2, 4, 4, 8), init="randn", seed=42)
_I4C4 = InputSpec(shape=(4, 4, 4, 4, 4), init="randn", seed=42)
_I2C4 = InputSpec(shape=(2, 4, 4, 4, 4), init="randn", seed=42)

_W4_222 = InputSpec(shape=(4, 2, 2, 2, 2), init="randn", seed=43)
_W4_422 = InputSpec(shape=(4, 4, 2, 2, 2), init="randn", seed=43)
_W2_422 = InputSpec(shape=(2, 4, 2, 2, 2), init="randn", seed=43)
_W4_111 = InputSpec(shape=(4, 2, 1, 1, 1), init="randn", seed=43)
_B4 = InputSpec(shape=(4,), init="randn", seed=44)


def _conv3d(x, w):
    return F.conv3d(x, w)


def _conv3d_with_bias(x, w, b):
    return F.conv3d(x, w, b)


def _conv3d_groups(x, w):
    return F.conv3d(x, w, groups=2)


def _conv3d_groups_with_bias(x, w, b):
    return F.conv3d(x, w, bias=b, groups=2)


register(OpShardCase(
    name="conv3d_ops_data_parallel",
    fn=_conv3d,
    inputs=[_I4, _W4_222],
    placements=[(Shard(0), Replicate(), Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_column_parallel",
    fn=_conv3d,
    inputs=[_I4B4, _W4_222],
    placements=[(Replicate(), Replicate(), Replicate(), Replicate(), Replicate()),
                (Shard(0), Replicate(), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_spatial_parallel",
    fn=_conv3d,
    inputs=[_I4H8, _W4_222],
    placements=[(Replicate(), Replicate(), Shard(0), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_with_bias",
    fn=_conv3d_with_bias,
    inputs=[_I4B4, _W4_222, _B4],
    placements=[(Replicate(), Replicate(), Replicate(), Replicate(), Replicate()),
                (Shard(0), Replicate(), Replicate(), Replicate(), Replicate()),
                (Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_row_parallel",
    fn=_conv3d,
    inputs=[_I5H4, _W4_422],
    placements=[(Replicate(), Shard(1), Replicate(), Replicate(), Replicate()),
                (Replicate(), Shard(1), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_dp_cp",
    fn=_conv3d,
    inputs=[_I4B4, _W4_222],
    placements=[(Shard(0), Replicate(), Replicate(), Replicate(), Replicate()),
                (Replicate(), Shard(0), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_dp_rp",
    fn=_conv3d,
    inputs=[_I5H4, _W2_422],
    placements=[(Shard(0), Shard(1), Replicate(), Replicate(), Replicate()),
                (Replicate(), Shard(1), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_spatial_h",
    fn=_conv3d,
    inputs=[_I4H8, _W4_111],
    placements=[(Replicate(), Shard(3)),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_spatial_w",
    fn=_conv3d,
    inputs=[_I4W8, _W4_111],
    placements=[(Replicate(), Shard(4)),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_groups_dp",
    fn=_conv3d_groups,
    inputs=[_I4C4, _W4_222],
    placements=[(Shard(0), Replicate(), Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_groups_cp",
    fn=_conv3d_groups,
    inputs=[_I2C4, _W4_222],
    placements=[(Replicate(), Replicate(), Replicate(), Replicate(), Replicate()),
                (Replicate(), Shard(0), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv3d_ops_groups_cp_with_bias",
    fn=_conv3d_groups_with_bias,
    inputs=[_I2C4, _W4_222, _B4],
    placements=[(Replicate(), Replicate(), Replicate(), Replicate(), Replicate()),
                (Replicate(), Shard(0), Replicate(), Replicate(), Replicate()),
                (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))
