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
"""Shard ops cases for Partial-aware ``torch.add`` and ``torch.sub``."""
import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from tests.shard_ops.framework import CompareSpec, InputSpec, OpShardCase, register


def _partial_matmul(x, w):
    """Produce a Partial(sum) tensor by sharding the contracting dimension."""
    return torch.matmul(x, w)


def _add_partial_replicate(x, w, bias):
    return torch.add(_partial_matmul(x, w), bias)


def _sub_replicate_partial(x, w, bias):
    return torch.sub(bias, _partial_matmul(x, w))


def _collect_backward_grads(x, w, bias, op):
    """Return effective global gradients for a local-output backward pass."""
    is_distributed = isinstance(x, DTensor)
    if is_distributed:
        local_inputs = (x.to_local(), w.to_local(), bias.to_local())
        for local_input in local_inputs:
            local_input.requires_grad_(True)
            local_input.retain_grad()
    else:
        local_inputs = (x, w, bias)
        for local_input in local_inputs:
            local_input.requires_grad_(True)

    output = op(x, w, bias)
    loss = output.to_local().sum() if is_distributed else output.sum()
    loss.backward()

    grads = tuple(local_input.grad for local_input in local_inputs)
    if any(grad is None for grad in grads):
        raise AssertionError("add/sub backward did not populate every input gradient")
    if not is_distributed:
        return tuple(grad + 0 for grad in grads)

    mesh = x.device_mesh
    return (
        DTensor.from_local(grads[0], mesh, x.placements),
        DTensor.from_local(grads[1], mesh, (Partial("sum"), Shard(0))),
        DTensor.from_local(grads[2], mesh, (Shard(0), Partial("sum"))),
    )


def _add_partial_replicate_backward(x, w, bias):
    return _collect_backward_grads(x, w, bias, _add_partial_replicate)


def _sub_replicate_partial_backward(x, w, bias):
    return _collect_backward_grads(x, w, bias, _sub_replicate_partial)


_INPUTS = [
    InputSpec(shape=(8, 4), init="randn", seed=41),
    InputSpec(shape=(4, 6), init="randn", seed=42),
    InputSpec(shape=(8, 6), init="randn", seed=43),
]
_PLACEMENTS = [
    (Shard(0), Shard(1)),
    (Replicate(), Shard(0)),
    (Shard(0), Replicate()),
]


register(OpShardCase(
    name="add_ops_partial_replicate",
    fn=_add_partial_replicate,
    inputs=_INPUTS,
    placements=_PLACEMENTS,
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))


register(OpShardCase(
    name="sub_ops_replicate_partial",
    fn=_sub_replicate_partial,
    inputs=_INPUTS,
    placements=_PLACEMENTS,
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))


register(OpShardCase(
    name="add_ops_partial_replicate_backward",
    fn=_add_partial_replicate_backward,
    inputs=_INPUTS,
    placements=_PLACEMENTS,
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level1", "npu_level1"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))


register(OpShardCase(
    name="sub_ops_replicate_partial_backward",
    fn=_sub_replicate_partial_backward,
    inputs=_INPUTS,
    placements=_PLACEMENTS,
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level1", "npu_level1"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))
