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
"""Shard ops cases for ``torch.nn.functional.conv1d``."""
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

_I4C2 = InputSpec(shape=(4, 2, 8), init="randn", seed=42)
_I2C4 = InputSpec(shape=(2, 4, 8), init="randn", seed=52)
_I4C4 = InputSpec(shape=(4, 4, 8), init="randn", seed=62)

_W4C2 = InputSpec(shape=(4, 2, 3), init="randn", seed=43)
_W2C4 = InputSpec(shape=(2, 4, 3), init="randn", seed=53)
_W4C2K5 = InputSpec(shape=(4, 2, 5), init="randn", seed=63)
_W4C1 = InputSpec(shape=(4, 1, 3), init="randn", seed=73)
_B4 = InputSpec(shape=(4,), init="randn", seed=44)


def _call_operator(func, *args, **kwargs):
    return func(*args, **kwargs)


def _conv1d(x, w):
    return _call_operator(F.conv1d, x, w)


def _conv1d_with_bias(x, w, b):
    return _call_operator(F.conv1d, x, w, b)


def _conv1d_stride(x, w, b):
    return _call_operator(F.conv1d, x, w, b, stride=2, padding=1)


def _conv1d_dilation(x, w, b):
    return _call_operator(F.conv1d, x, w, b, dilation=2, padding=5)


def _conv1d_same(x, w, b):
    return _call_operator(F.conv1d, x, w, b, padding="same")


def _conv1d_groups_with_bias(x, w, b):
    return _call_operator(F.conv1d, x, w, bias=b, groups=2)


def _conv1d_depthwise_with_bias(x, w, b):
    return _call_operator(F.conv1d, x, w, bias=b, groups=4)


register(OpShardCase(
    name="conv1d_ops_replicated",
    fn=_conv1d,
    inputs=[_I4C2, _W4C2],
    placements=[(Replicate(), Replicate()),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_replicated_with_bias",
    fn=_conv1d_with_bias,
    inputs=[_I4C2, _W4C2, _B4],
    placements=[(Replicate(), Replicate()),
                (Replicate(), Replicate()),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_data_parallel",
    fn=_conv1d,
    inputs=[_I4C2, _W4C2],
    placements=[(Shard(0), Replicate()),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_column_parallel_with_bias",
    fn=_conv1d_with_bias,
    inputs=[_I4C2, _W4C2, _B4],
    placements=[(Replicate(), Replicate()),
                (Replicate(), Shard(0)),
                (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_column_parallel",
    fn=_conv1d,
    inputs=[_I4C2, _W4C2],
    placements=[(Replicate(), Replicate()),
                (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_row_parallel",
    fn=_conv1d,
    inputs=[_I2C4, _W2C4],
    placements=[(Replicate(), Shard(1)),
                (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_dp_cp",
    fn=_conv1d_with_bias,
    inputs=[_I4C2, _W4C2, _B4],
    placements=[(Shard(0), Replicate()),
                (Replicate(), Shard(0)),
                (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_groups_cp_with_bias",
    fn=_conv1d_groups_with_bias,
    inputs=[_I2C4, _W4C2, _B4],
    placements=[(Replicate(),),
                (Shard(0),),
                (Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_depthwise_group_aligned",
    fn=_conv1d_depthwise_with_bias,
    inputs=[_I2C4, _W4C1, _B4],
    placements=[(Shard(1),),
                (Shard(0),),
                (Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_groups_dp",
    fn=_conv1d_groups_with_bias,
    inputs=[_I4C4, _W4C2, _B4],
    placements=[(Shard(0), Replicate()),
                (Replicate(), Replicate()),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_groups_dp_cp_with_bias",
    fn=_conv1d_groups_with_bias,
    inputs=[_I4C4, _W4C2, _B4],
    placements=[(Shard(0), Replicate()),
                (Replicate(), Shard(0)),
                (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_non_default_stride",
    fn=_conv1d_stride,
    inputs=[_I4C2, _W4C2, _B4],
    placements=[(Replicate(), Replicate()),
                (Replicate(), Replicate()),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_dilation",
    fn=_conv1d_dilation,
    inputs=[_I4C2, _W4C2K5, _B4],
    placements=[(Replicate(), Replicate()),
                (Replicate(), Replicate()),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="conv1d_ops_str_padding_same",
    fn=_conv1d_same,
    inputs=[_I4C2, _W4C2, _B4],
    placements=[(Replicate(), Replicate()),
                (Replicate(), Replicate()),
                (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))
