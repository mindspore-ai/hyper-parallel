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
"""parallel_pad test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_pad import PadDistributedOp

op = PadDistributedOp("pad")


def test_pad_infer_layout_success_unsharded():
    """
    Feature: Pad unsharded dimension
    Description: Pad the last dimension which is Replicated.
    Expectation: Output layout is identical to input layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: Shard(0) on dim0, Replicate on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Pad dim1 (last dim): (1, 1)
    # extra_args[0] is pad tuple
    pad = (1, 1)

    output_layout = op.infer_layout((x_layout,), extra_args=(pad,))

    # Validate result is same object or equivalent map
    assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]


def test_pad_infer_layout_fail_sharded():
    """
    Feature: Pad sharded dimension
    Description: Attempt to pad a dimension that is Sharded with non-zero values.
    Expectation: Raises ValueError.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: Shard(0) on dim0, Replicate on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Pad dim0 (2nd last) with (1, 1) and dim1 (last) with (0, 0)
    # Pad tuple structure: (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
    pad = (0, 0, 1, 1)

    with pytest.raises(ValueError, match="does not support padding on a sharded dimension"):
        op.infer_layout((x_layout,), extra_args=(pad,))


def test_pad_infer_layout_mixed_dims():
    """
    Feature: Pad specific dimension in multi-dim tensor
    Description: Pad only the replicated dimension in a 3D tensor with mixed sharding.
    Expectation: Success for unsharded pad, Failure for sharded pad.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # 3D Tensor: dim0=Shard(0), dim1=Shard(1), dim2=Replicate
    x_placements = (Shard(0), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    # 1. Pad dim2 (Replicate) only: (1, 1) -> Success
    pad_success = (1, 1)
    output_layout = op.infer_layout((x_layout,), extra_args=(pad_success,))
    assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]

    # 2. Pad dim1 (Shard) with non-zero: (0, 0, 1, 1) -> Fail
    pad_fail = (0, 0, 1, 1)
    with pytest.raises(ValueError, match="does not support padding on a sharded dimension"):
        op.infer_layout((x_layout,), extra_args=(pad_fail,))


def test_pad_infer_layout_zero_padding_on_sharded():
    """
    Feature: Zero padding on sharded dimension
    Description: If pad values are 0, it should be allowed even if dimension is sharded.
    Expectation: Success.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # dim0 sharded
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Pad dim0 with (0,0) and dim1 with (1,1): (1, 1, 0, 0)
    # dim1 is Replicate (valid to pad), dim0 is Shard (valid since pad is 0)
    pad = (1, 1, 0, 0)

    output_layout = op.infer_layout((x_layout,), extra_args=(pad,))
    assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]


def test_pad_infer_layout_invalid_args():
    """
    Feature: Validate arguments
    Description: Check invalid pad tuple length or size mismatches.
    Expectation: Raises ValueError.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(),)
    x_layout = _build_layout(mesh, x_placements, 1)

    # Case 1: Odd length pad tuple
    with pytest.raises(ValueError, match="Pad tuple length must be even"):
        op.infer_layout((x_layout,), extra_args=((1, 2, 3),))

    # Case 2: Padding more dims than tensor has
    with pytest.raises(ValueError, match="but tensor only has"):
        op.infer_layout((x_layout,), extra_args=((1, 1, 1, 1),))
