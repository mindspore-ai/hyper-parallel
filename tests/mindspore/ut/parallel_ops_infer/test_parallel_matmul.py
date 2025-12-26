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
# ============================================================================
"""parallel_matmul test"""

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_matmul import MatMulDistributedOp

op = MatMulDistributedOp("MatMul")


def test_matmul_layout_data_parallel():
    """
    Feature: MatMul data parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("None", "None")

    output_layout = op.infer_layout((x_layout, w_layout), (False, True))
    expected_map = (1, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


def test_matmul_layout_hybrid_parallel():
    """
    Feature: MatMul hybrid parallel
    Description: Hybrid parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Hybrid Parallel (DP + MP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("mp", "None")

    output_layout = op.infer_layout((x_layout, w_layout), (False, True))
    expected_map = (1, 0)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Hybrid Parallel test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_matmul_layout_tensor_parallel():
    """
    Feature: MatMul tensor parallel
    Description: Tensor parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Tensor Parallel (TMP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("mp", "None")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("mp", "None")

    output_layout = op.infer_layout((x_layout, w_layout), (True, False))
    expected_map = (-1, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Tensor Parallel test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_matmul_layout_hybrid_tensor_parallel():
    """
    Feature: MatMul hybrid tensor parallel
    Description: Hybrid tensor parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Hybrid Tensor Parallel
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("mp", "dp")
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("None", "mp")

    # Test without transpose
    output_layout = op.infer_layout((x_layout, w_layout), (True, True))
    expected_map = (1, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Hybrid Tensor Parallel test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_matmul_layout_multi_shard_tensor_parallel():
    """
    Feature: MatMul multi shard tensor parallel
    Description: Multi shard tensor tensor parallel scenario
    Expectation: Success
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "tp", "mp")
    rank_list = list(range(8))

    # Multi-Shard Tensor Parallel
    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "mp")
    w_layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = w_layout("tp", "mp")

    # Test without transpose
    output_layout = op.infer_layout((x_layout, w_layout), (False, True))
    expected_map = (2, 1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Multi-Shard Tensor Parallel test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_matmul_layout_multi_shard_one_dim_tensor_parallel():
    """
    Feature: MatMul multi shard one dim tensor parallel
    Description: Multi shard one dim tensor tensor parallel scenario
    Expectation: Success
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "tp", "mp")
    rank_list = list(range(8))

    # Multi-Shard One Dimension Tensor Parallel
    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout(("dp", "tp"), "None")
    w_layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = w_layout("None", "mp")

    # Test without transpose
    output_layout = op.infer_layout((x_layout, w_layout), (False, False))
    expected_map = ((2, 1), 0)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Multi-Shard One Dim Tensor Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"

    # Multi-Shard One Dimension Tensor Parallel
    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout(("dp", "tp"), "mp")
    w_layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = w_layout("None", "mp")

    # Test without transpose
    output_layout = op.infer_layout((x_layout, w_layout), (False, True))
    expected_map = ((2, 1), -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Multi-Shard One Dim Tensor Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"

    # Multi-Shard One Dimension Tensor Parallel
    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout(("dp", "tp"), "mp")
    w_layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = w_layout(("dp", "tp"), "None")

    # Test without transpose
    output_layout = op.infer_layout((x_layout, w_layout), (True, False))
    expected_map = (0, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Multi-Shard One Dim Tensor Parallel test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"
