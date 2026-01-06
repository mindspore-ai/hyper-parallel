# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test ut for parallel arithmetic"""
import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_elementwise import ElementWiseDistributedOp


def test_add_layout_hybrid_parallel():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ElementWiseDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "mp")

    extra_args = {"input_shapes": [(4, 16), (4, 16)]}
    output_layout = op.infer_layout((x_layout, w_layout), (extra_args))
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_add_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ElementWiseDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    extra_args = {"input_shapes": [(4, 16), (4, 16)]}
    output_layout = op.infer_layout((x_layout, w_layout), (extra_args))
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_mul_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ElementWiseDistributedOp("Mul")

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    extra_args = {"input_shapes": [(4, 16), (4, 16)]}
    output_layout = op.infer_layout((x_layout, w_layout), extra_args)
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_sub_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ElementWiseDistributedOp("Sub")

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    extra_args = {"input_shapes": [(4, 16), (4, 16)]}
    output_layout = op.infer_layout((x_layout, w_layout), extra_args)
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_div_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ElementWiseDistributedOp("Div")

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    extra_args = {"input_shapes": [(4, 16), (4, 16)]}
    output_layout = op.infer_layout((x_layout, w_layout), extra_args)
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_tuple_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))
    op = ElementWiseDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout(("dp", "mp"), "cp")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout(("dp", "mp"), "None")
    extra_args = {"input_shapes": [(4, 16), (4, 16)]}
    output_layout = op.infer_layout((x_layout, w_layout), extra_args)
    print(output_layout.alias_tensor_map)
    expected_map = (("dp", "mp"), "cp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_multislice_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 1, 4)
    base_alias_name = ("dp", "cp", "tp")
    base_rank_list = list(range(8))
    op = ElementWiseDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("cp", "dp", "None")
    print("x_layout", x_layout)
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout(('cp', 'tp'), "dp", "None")
    print("w_layout", w_layout)
    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}
    with pytest.raises(ValueError):
        _ = op.infer_layout((x_layout, w_layout), extra_args)
