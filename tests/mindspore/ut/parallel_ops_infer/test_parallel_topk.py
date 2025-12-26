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
"""parallel_topk test"""

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_topk import TopKDistributedOp

op = TopKDistributedOp("TopK")


def test_topk_layout_data_parallel():
    """
    Feature: TopK data parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Data Parallel
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    values_layout, indices_layout = op.infer_layout((x_layout,), True)
    expected_map = (1, -1)  # Expected output tensor map
    assert values_layout.to_dict()["tensor_map"] == indices_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {values_layout.to_dict()['tensor_map']}"


def test_topk_layout_tensor_parallel():
    """
    Feature: TopK tensor parallel
    Description: Tensor parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Tensor Parallel
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("mp", "None")

    values_layout, indices_layout = op.infer_layout((x_layout,), True)
    expected_map = (0, -1)  # Expected output tensor map
    assert values_layout.to_dict()["tensor_map"] == indices_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {values_layout.to_dict()['tensor_map']}"


def test_topk_layout_tensor_and_data_parallel():
    """
    Feature: TopK tensor parallel
    Description: Tensor parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Data and Tensor Parallel
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")

    values_layout, indices_layout = op.infer_layout((x_layout,), True)
    expected_map = (1, 0)  # Expected output tensor map
    assert values_layout.to_dict()["tensor_map"] == indices_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {values_layout.to_dict()['tensor_map']}"
