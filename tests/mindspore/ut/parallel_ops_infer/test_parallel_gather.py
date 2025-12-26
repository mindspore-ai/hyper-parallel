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
"""parallel_gather test"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_gather import GatherDistributedOp, IndexSelectDistributedOp


def run_scenario(scenario_name, op_name, params_layout, indices_layout, axis, expected_map=None):
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    op = GatherDistributedOp(op_name) if op_name == "GatherD" else IndexSelectDistributedOp(op_name)
    output_layout = op.infer_layout((params_layout, None, indices_layout), (axis,))
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test BatchMatMulExt failed. Expected {expected_map},"
        f" got {output_layout.to_dict()['tensor_map']}"
    )


def test_index_select_parallel_1():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    p_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    p_layout = p_layout("None", "a")

    i_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    i_layout = i_layout("b")

    run_scenario(
        "1. [index_select] Params shard: (1, 2), Indices shard: (4)",
        "IndexSelect",
        p_layout,
        i_layout,
        axis=0,
        expected_map=(0, 1),
    )


def test_index_select_parallel_2():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    p_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    p_layout = p_layout("b", "a")

    i_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    i_layout = i_layout("b")

    with pytest.raises(ValueError):
        run_scenario(
            "2. [index_select] Params shard: (4, 2), Indices shard: (4)",
            "IndexSelect",
            p_layout,
            i_layout,
            axis=0,
        )


def test_gather_parallel_1():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    p_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    p_layout = p_layout("None", "a")

    i_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    i_layout = i_layout("b", "a")

    run_scenario(
        "1. [gather] Params shard: (1, 2), Indices shard: (4, 2)",
        "GatherD",
        p_layout,
        i_layout,
        axis=0,
        expected_map=(0, 1),
    )
