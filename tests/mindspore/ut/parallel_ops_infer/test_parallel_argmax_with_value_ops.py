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

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_argmax_with_value_ops import ArgMaxWithValueDistributedOp


def run_scenario(scenario_name, tensor_layout, axis, keep_dims, expected_map):
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    op = ArgMaxWithValueDistributedOp("ArgMaxWithValue")
    output_layout, _ = op.infer_layout((tensor_layout, None, None), (axis, keep_dims))
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test ArgMaxWithValue failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


def test_parallel_1():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (2, 4, 1)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    tensor_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)("a", "None", "c")

    run_scenario(
        "ArgMaxWithValue Parallel 1",
        tensor_layout,
        axis=0,
        keep_dims=True,
        expected_map=(-1, -1, 0),
    )


def test_parallel_2():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (2, 4, 1)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    tensor_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)("a", "None", "c")

    run_scenario(
        "ArgMaxWithValue Parallel 1",
        tensor_layout,
        axis=0,
        keep_dims=False,
        expected_map=(-1, 0),
    )


def test_parallel_3():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (2, 4, 1)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    tensor_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)("a", ("None", "c"), "None")

    run_scenario(
        "ArgMaxWithValue Parallel 1",
        tensor_layout,
        axis=0,
        keep_dims=False,
        expected_map=((-1, 0), -1),
    )
