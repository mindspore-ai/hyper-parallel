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
"""parallel_concat test"""

from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_concat import ConcatDistributedOp


def run_scenario(scenario_name, tensor_1_layout, tensor_2_layout, dim, expected_map):
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    op = ConcatDistributedOp("Concat")
    output_layout = op.infer_layout((tensor_1_layout, tensor_2_layout), (dim,))
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test Concat failed. Expected {expected_map},"
        f" got {output_layout.to_dict()['tensor_map']}"
    )


def test_concat_parallel_1():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_mesh_shape = (2, 4, 1)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    t1_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "b", "c"
    )
    t2_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "None", "b", "c"
    )

    run_scenario(
        "Concat Parallel 1",
        t1_layout,
        t2_layout,
        dim=0,
        expected_map=(2, 1, 0),
    )


def test_concat_parallel_2():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_mesh_shape = (2, 4, 1)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    t1_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "b", "c"
    )
    t2_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "None", "c"
    )

    run_scenario(
        "Concat Parallel 2",
        t1_layout,
        t2_layout,
        dim=1,
        expected_map=(2, 1, 0),
    )


def test_concat_parallel_3():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_mesh_shape = (2, 4, 1)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    t1_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "b", "None"
    )
    t2_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "b", "None"
    )

    run_scenario(
        "Concat Parallel 3",
        t1_layout,
        t2_layout,
        dim=2,
        expected_map=(2, 1, -1),
    )
