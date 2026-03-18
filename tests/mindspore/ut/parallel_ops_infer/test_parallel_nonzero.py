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
"""parallel_nonzero test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_nonzero import NonzeroDistributedOp

op = NonzeroDistributedOp("nonzero")


def test_nonzero_layout_inference_basic():
    """
    Feature: Nonzero on fully replicated input (as_tuple=False)
    Description: Input is a fully replicated 2D tensor. Nonzero returns a single 2D tensor.
    Expectation: Output layout is a single fully replicated 2D layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input MUST be fully replicated
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Infer layout without extra_args (as_tuple defaults to False)
    output_layout = op.infer_layout((x_layout,))

    expected_map = (-1, -1)  # 2D output, fully unsharded
    assert not isinstance(output_layout, tuple)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Basic nonzero failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_nonzero_layout_inference_as_tuple():
    """
    Feature: Nonzero on fully replicated input (as_tuple=True)
    Description: Input is a fully replicated 3D tensor. Nonzero with as_tuple=True
                 returns a tuple of 1D tensors matching input ndim.
    Expectation: Output is a tuple of 3 fully replicated 1D layouts.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input MUST be fully replicated
    x_placements = (Replicate(), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    # Infer layout with as_tuple=True in extra_args
    output_layouts = op.infer_layout((x_layout,), extra_args=(True,))

    expected_map = (-1,)  # 1D output, fully unsharded

    assert isinstance(output_layouts, tuple), "Expected output to be a tuple of layouts"
    assert len(output_layouts) == 3, "Expected 3 layouts for a 3D input"

    for i, out_layout in enumerate(output_layouts):
        assert out_layout.to_dict()["tensor_map"] == expected_map, \
            f"Tuple output {i} layout mismatch. Expected {expected_map}, got {out_layout.to_dict()['tensor_map']}"


def test_nonzero_layout_invalid_sharded_input():
    """
    Feature: Nonzero on sharded input
    Description: Attempt to run nonzero on a tensor with a sharded dimension.
                 Since nonzero generates data-dependent dynamic shapes, it is unsafe.
    Expectation: ValueError raised with clear message.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input is sharded on dim0 ("dp"), which is invalid for nonzero
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to infer layout with sharded input
    with pytest.raises(ValueError, match="input tensor must be fully replicated"):
        op.infer_layout((x_layout,))


def test_nonzero_layout_invalid_partial_input():
    """
    Feature: Nonzero on partial input
    Description: Attempt to run nonzero on a tensor with a Partial state.
    Expectation: ValueError raised by the base class check.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )

    # Input has a partial status on dim0
    x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    # Manually inject a partial state for testing the guardrail
    x_layout.set_partial_by_dev_axis("dp", "sum")

    # Attempt to infer layout with partial input
    with pytest.raises(ValueError, match="has Partial status which is not allowed"):
        op.infer_layout((x_layout,))
