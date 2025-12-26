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
"""parallel_softmax test"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_activation_with_axis import ActivationWithAxisDistributedOp

op = ActivationWithAxisDistributedOp("Softmax")


def test_softmax_layout_data_parallel():
    """
    Feature: Softmax data parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    output_layout = op.infer_layout((x_layout,), (-1,))
    expected_map = (0, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Data Parallel with transpose_a test failed. Expected {expected_map},"
        f" got {output_layout.to_dict()['tensor_map']}"
    )


def test_softmax_layout_data_parallel_value_failed():
    """
    Feature: Softmax data parallel value failed
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    with pytest.raises(ValueError):
        _ = op.infer_layout((x_layout,), (0,))
