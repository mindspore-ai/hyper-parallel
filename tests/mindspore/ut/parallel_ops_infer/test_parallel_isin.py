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
"""parallel_isin test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_isin import IsinDistributedOp

# 创建 isin 算子实例
op = IsinDistributedOp("isin")


def test_isin_layout_data_parallel():
    """
    Feature: Isin data parallel
    Description: elements sharded on data parallel dimension, test_elements fully replicated
    Expectation: Output layout identical to elements layout
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    # elements: sharded on dim0 ("dp"), unsharded on dim1
    elements_placements = (Shard(0), Replicate())
    elements_layout = _build_layout(mesh, elements_placements, 2)  # tensor_map = (1, -1)

    # test_elements: fully replicated (critical constraint)
    test_elements_placements = (Replicate(), Replicate())
    test_elements_layout = _build_layout(mesh, test_elements_placements, 2) # tensor_map = (-1, -1)

    output_layout = op.infer_layout((elements_layout, test_elements_layout), extra_args=None)

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data parallel isin failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_isin_layout_mixed_parallel():
    """
    Feature: Isin mixed parallel
    Description: elements with mixed sharding (dp on dim0, mp on dim2), test_elements fully replicated
    Expectation: Output layout identical to elements layout
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 3, 4),
        alias_name = ("dp", "tp", "mp")
    )

    # elements: sharded on dim0 ("dp"), unsharded on dim1, sharded on dim2 ("mp")
    elements_placements = (Shard(0), Replicate(), Shard(2))
    elements_layout = _build_layout(mesh, elements_placements, 3)  # tensor_map = (2, -1, 0)

    # test_elements: fully replicated (1D tensor in this test)
    test_elements_placements = (Replicate(), Replicate(), Replicate())
    test_elements_layout = _build_layout(mesh, test_elements_placements, 3)

    output_layout = op.infer_layout((elements_layout, test_elements_layout), extra_args=None)

    expected_map = (2, -1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Mixed parallel isin failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_isin_layout_invalid_test_elements_sharded():
    """
    Feature: Isin with test_elements sharded on dim0
    Description: test_elements sharded on first dimension violates global view requirement
    Expectation: ValueError raised with clear message
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    # elements: fully replicated (valid)
    elements_placements = (Replicate(), Replicate())
    elements_layout = _build_layout(mesh, elements_placements, 2)

    # test_elements: sharded on dim0 (INVALID - violates global view requirement)
    test_elements_placements = (Shard(0), Replicate())
    test_elements_layout = _build_layout(mesh, test_elements_placements, 2)  # tensor_map = (1, -1)

    with pytest.raises(ValueError, match="'test_elements' must be unsharded. Current tensor_map:"):
        op.infer_layout((elements_layout, test_elements_layout), extra_args=None)


def test_isin_layout_missing_test_elements():
    """
    Feature: Isin without test_elements layout
    Description: Missing second layout in tuple
    Expectation: ValueError raised
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    elements_placements = (Replicate(), Replicate())
    elements_layout = _build_layout(mesh, elements_placements, 2)

    # Only one layout provided (missing test_elements)
    with pytest.raises(ValueError, match="'test_elements' requires a valid tensor layout"):
        op.infer_layout((elements_layout,), extra_args=None)

    # test_elements layout is None
    with pytest.raises(ValueError, match="'test_elements' requires a valid tensor layout"):
        op.infer_layout((elements_layout, None), extra_args=None)
