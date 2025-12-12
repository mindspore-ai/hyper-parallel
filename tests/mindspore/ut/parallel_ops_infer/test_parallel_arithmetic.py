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
from hyper_parallel.core.tensor_parallel.ops.parallel_arithmetic import ArithmeticDistributedOp


def test_add_layout_hybrid_parallel():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ArithmeticDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "mp")

    output_layout = op.infer_layout((x_layout, w_layout), ())
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_add_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ArithmeticDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    output_layout = op.infer_layout((x_layout, w_layout), ())
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_mul_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ArithmeticDistributedOp("Mul")

    # Data Parallel (DP)
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    output_layout = op.infer_layout((x_layout, w_layout), ())
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_sub_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ArithmeticDistributedOp("Sub")

    # Data Parallel (DP)
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    output_layout = op.infer_layout((x_layout, w_layout), ())
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_div_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    op = ArithmeticDistributedOp("Div")

    # Data Parallel (DP)
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("dp", "None")
    output_layout = op.infer_layout((x_layout, w_layout), ())
    print(output_layout.alias_tensor_map)
    expected_map = ("dp", "mp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_tuple_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_device_matrix = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))
    op = ArithmeticDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout(("dp", "mp"), "cp")
    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout(("dp", "mp"), "None")
    output_layout = op.infer_layout((x_layout, w_layout), ())
    print(output_layout.alias_tensor_map)
    expected_map = (("dp", "mp"), "cp")  # Expected output tensor map
    assert output_layout.alias_tensor_map == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.alias_tensor_map}"


def test_multislice_layout_broadcast():
    """
    Feature: add hybrid parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_device_matrix = (2, 1, 4)
    base_alias_name = ("dp", "cp", "tp")
    base_rank_list = list(range(8))
    op = ArithmeticDistributedOp("Add")

    # Data Parallel (DP)
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("cp", "dp", "None")
    print("x_layout", x_layout)
    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout(('cp', 'tp'), "dp", "None")
    print("w_layout", w_layout)
    with pytest.raises(ValueError):
        _ = op.infer_layout((x_layout, w_layout), ())
