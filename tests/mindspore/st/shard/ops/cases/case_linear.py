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

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
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


def _linear_with_bias_backward(x, w, b):
    """Return effective global gradients from a local row-parallel backward."""
    is_distributed = isinstance(x, DTensor)
    if is_distributed:
        local_inputs = (x.to_local(), w.to_local(), b.to_local())
    else:
        local_inputs = (x, w, b)
    for local_input in local_inputs:
        local_input.requires_grad = True
        if is_distributed:
            local_input.retain_grad()

    output = _linear_with_bias(x, w, b)
    loss = output.to_local().sum() if is_distributed else output.sum()
    loss.backward()

    grads = tuple(local_input.grad for local_input in local_inputs)
    if any(grad is None for grad in grads):
        raise AssertionError("linear backward did not populate every input gradient")
    if not is_distributed:
        return tuple(grad + 0 for grad in grads)

    mesh = x.device_mesh
    return (
        DTensor.from_local(grads[0], mesh, x.placements),
        DTensor.from_local(grads[1], mesh, (Partial("sum"), Shard(1))),
        DTensor.from_local(grads[2], mesh, (Partial("sum"), Partial("sum"))),
    )


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


register(OpShardCase(
    name="linear_ops_tp_row_with_bias",
    fn=_linear_with_bias,
    inputs=[_X_SPEC, _W_SPEC, _B_SPEC],
    placements=[
        (Shard(0), Shard(1)),          # x: batch on dp, in_features on tp
        (Replicate(), Shard(1)),       # w: in_features on tp
        (Replicate(), Replicate()),    # b: one contribution per tp group
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="linear_ops_tp_row_with_bias_backward",
    fn=_linear_with_bias_backward,
    inputs=[_X_SPEC, _W_SPEC, _B_SPEC],
    placements=[
        (Shard(0), Shard(1)),
        (Replicate(), Shard(1)),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
