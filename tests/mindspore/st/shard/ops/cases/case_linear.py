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
"""Shard ops cases for MindSpore ``mint.nn.functional.linear``.

Uses ``ms.mint.nn.functional.linear`` exclusively per project convention
(no ``ops.MatMul``). Matches the torch ``F.linear`` semantics:
``y = x @ w.T + b``.
"""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _linear_no_bias(x, w):
    return ms.mint.nn.functional.linear(x, w)


def _linear_with_bias(x, w, b):
    return ms.mint.nn.functional.linear(x, w, b)


_X_SPEC = InputSpec(shape=(16, 8), init="randn", seed=11)
_W_SPEC = InputSpec(shape=(4, 8), init="randn", seed=12)
_B_SPEC = InputSpec(shape=(4,), init="randn", seed=13)


register(OpShardCase(
    name="linear_ops_dp_no_bias",
    fn=_linear_no_bias,
    inputs=[_X_SPEC, _W_SPEC],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="linear_ops_dp_with_bias",
    fn=_linear_with_bias,
    inputs=[_X_SPEC, _W_SPEC, _B_SPEC],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="linear_ops_tp_col_with_bias",
    fn=_linear_with_bias,
    inputs=[_X_SPEC, _W_SPEC, _B_SPEC],
    # Column-TP: shard out_features (tensor dim 0 of ``w``/``b``) on the
    # ``tp`` mesh axis. Placement tuples are indexed per mesh dim;
    # ``Shard(N)`` references tensor dim ``N``.
    placements=[
        (Shard(0), Replicate()),       # x: batch on dp
        (Replicate(), Shard(0)),       # w: out_features on tp
        (Replicate(), Shard(0)),       # b: out_features on tp
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
