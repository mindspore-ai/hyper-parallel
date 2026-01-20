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
"""parallel_linear test"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_matmul import LinearDistributedOp

op = LinearDistributedOp("Linear")

base_mesh_shape = (2, 4)
base_alias_name = ("dp", "mp")
base_rank_list = list(range(8))
layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

def test_linear_layout_data_parallel():
    """
    Feature: Linear data parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    # Data Parallel (DP)
    x_layout = layout("dp", "None")
    w_layout = layout("None", "None")
    output_layout = op.infer_layout((x_layout, w_layout, None), ())
    expected_map = (1, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with transpose_a test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


def test_linear_layout_hybrid_parallel():
    """
    Feature: Linear hybrid parallel
    Description: Hybrid parallel scenario
    Expectation: Success
    """
    # Hybrid Parallel (DP + MP)
    x_layout = layout("dp", "None")
    w_layout = layout("mp", "None")
    bias_layout = layout("mp")
    output_layout = op.infer_layout((x_layout, w_layout, bias_layout), ())
    expected_map = (1, 0)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Hybrid Parallel test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_linear_layout_hybrid_tensor_parallel():
    """
    Feature: Linear hybrid tensor parallel
    Description: Hybrid tensor parallel scenario
    Expectation: Success
    """
    # Hybrid Tensor Parallel
    x_layout = layout("dp", "mp")
    w_layout = layout("None", "mp")

    # Test without transpose
    output_layout = op.infer_layout((x_layout, w_layout, None), ())
    expected_map = (1, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Hybrid Tensor Parallel test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"

def test_linear_layout_hybrid_tensor_parallel_with_bias():
    """
    Feature: Linear hybrid tensor parallel
    Description: Hybrid tensor parallel scenario
    Expectation: raise error
    """
    # Hybrid Tensor Parallel
    x_layout = layout("dp", "mp")
    w_layout = layout("None", "mp")
    bias_layout = layout("None")
    with pytest.raises(ValueError):
        _ = op.infer_layout((x_layout, w_layout, bias_layout), ())


def test_linear_layout_partial_with_sharded_contract_dim():
    """
    Feature: Linear partial status with sharded contract dimension
    Description: Test that partial status is set when contract dimension is sharded
    Expectation: Success
    """

    # Contract dimension is sharded
    x_layout = layout("dp", "mp")
    w_layout = layout("None", "mp")
    output_layout = op.infer_layout((x_layout, w_layout, None), ())

    # Check that partial status is set for mp axis
    expected_partial = [None, 'sum']
    assert output_layout.partial == expected_partial, \
        f"Partial status test failed. Expected {expected_partial}, got {output_layout.partial}"


def test_linear_layout_partial_without_sharded_contract_dim():
    """
    Feature: Linear partial status without sharded contract dimension
    Description: Test that partial status is None when contract dimension is not sharded
    Expectation: Success
    """
    # Contract dimension is not sharded
    x_layout = layout("dp", "None")
    w_layout = layout("None", "mp")
    output_layout = op.infer_layout((x_layout, w_layout, None), ())

    # Check that partial status is None
    expected_partial = [None, None]
    assert output_layout.partial == expected_partial, \
        f"Partial status test failed. Expected {expected_partial}, got {output_layout.partial}"
