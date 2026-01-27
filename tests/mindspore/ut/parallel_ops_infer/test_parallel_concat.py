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
    """
    Runs a test scenario to verify the ConcatDistributedOp layout inference.

    Args:
        scenario_name (str): Name of the test scenario.
        tensor_1_layout (Layout): Layout object for the first tensor.
        tensor_2_layout (Layout): Layout object for the second tensor.
        dim (tuple): The dimension along which to concatenate.
        expected_map (dict): The expected tensor map dictionary.
    """
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    op = ConcatDistributedOp("Concat")
    # infer_layout returns a tuple of layouts, take the first one
    output_layout_tuple = op.infer_layout((tensor_1_layout, tensor_2_layout), (dim,))
    output_layout = output_layout_tuple[0]

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

def run_cat_scenario(scenario_name, tensor_1_layout, tensor_2_layout, dim, expected_map):
    """
    Runs a test scenario to verify the ConcatDistributedOp layout inference.

    Args:
        scenario_name (str): Name of the test scenario.
        tensor_1_layout (Layout): Layout object for the first tensor.
        tensor_2_layout (Layout): Layout object for the second tensor.
        dim (tuple): The dimension along which to concatenate.
        expected_map (dict): The expected tensor map dictionary.
    """
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    op = ConcatDistributedOp("cat")
    # infer_layout returns a tuple of layouts, take the first one
    output_layout_tuple = op.infer_layout((tensor_1_layout, tensor_2_layout), (dim,))
    output_layout = output_layout_tuple[0]

    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test cat failed. Expected {expected_map}, "
        f"got {output_layout.to_dict()['tensor_map']}"
    )


def test_cat_parallel_1():
    """
    Feature: PyTorch cat distributed op
    Description: concat on dim=0, shard on all dims
    Expectation: layout inherited from first input
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

    run_cat_scenario(
        "cat Parallel 1",
        t1_layout,
        t2_layout,
        dim=0,
        expected_map=(2, 1, 0),
    )


def test_cat_parallel_2():
    """
    Feature: PyTorch cat distributed op
    Description: concat on dim=1, different shard on concat dim
    Expectation: layout follows base_layout
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

    run_cat_scenario(
        "cat Parallel 2",
        t1_layout,
        t2_layout,
        dim=1,
        expected_map=(2, 1, 0),
    )


def test_cat_parallel_3():
    """
    Feature: PyTorch cat distributed op
    Description: concat on last dim, last dim replicated
    Expectation: tensor_map keeps -1 on concat dim
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

    run_cat_scenario(
        "cat Parallel 3",
        t1_layout,
        t2_layout,
        dim=2,
        expected_map=(2, 1, -1),
    )
def test_cat_parallel_4():
    """
    Feature: PyTorch cat distributed op
    Description: concat on dim=0, all inputs replicated
    Expectation: output is fully replicated
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    t1_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "None", "None"
    )
    t2_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "None", "None"
    )

    run_cat_scenario(
        "cat Parallel 4 (all replicated)",
        t1_layout,
        t2_layout,
        dim=0,
        expected_map=(-1, -1),
    )


def test_cat_parallel_5():
    """
    Feature: PyTorch cat distributed op
    Description: concat on dim=1, non-concat dim sharded
    Expectation: shard on non-concat dim is preserved
    """
    base_mesh_shape = (2, 2)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(4))

    t1_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "None"
    )
    t2_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "None"
    )

    run_cat_scenario(
        "cat Parallel 5 (shard on non-concat dim)",
        t1_layout,
        t2_layout,
        dim=1,
        expected_map=(1, -1),
    )


def test_cat_parallel_6():
    """
    Feature: PyTorch cat distributed op
    Description: 4D tensor concat on middle dim
    Expectation: tensor_map inferred correctly with reversed index
    """
    base_mesh_shape = (2, 2, 2)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    t1_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "b", "None", "c"
    )
    t2_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)(
        "a", "b", "None", "c"
    )

    run_cat_scenario(
        "cat Parallel 6 (4D middle dim)",
        t1_layout,
        t2_layout,
        dim=2,
        expected_map=(2, 1, -1, 0),
    )
