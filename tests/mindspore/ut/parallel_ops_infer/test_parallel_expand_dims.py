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
"""parallel_expand_dims test"""

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_expand_dims import ExpandDimsDistributedOp


def run_scenario(scenario_name, x_layout, expected_map, extra_args):
    """Infer layout of ExpandDims operator"""
    print(f"\n{'=' * 80}")
    print(f"Test ExpandDims, Scenario: {scenario_name}")
    print('=' * 80)

    op = ExpandDimsDistributedOp("ExpandDims")
    output_layout = op.infer_layout((x_layout,), extra_args)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"ExpandDims failed in scenario '{scenario_name}'. " \
        f"Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_expanddims_data_parallel_1():
    """
    Feature: Data parallel.
    Description: insert dimension at beginning, axis=0.
    Expectation: new dimension inserted at position 0.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")

    run_scenario(
        "1. Data Parallel (DP)",
        x_layout,
        expected_map=(-1, 2, -1, -1),
        extra_args=[0]
    )


def test_expanddims_model_parallel_2():
    """
    Feature: Model parallel.
    Description: insert dimension before mp axis, axis=2.
    Expectation: new dimension at position 2, mp shifted to position 3.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "mp")

    run_scenario(
        "2. Model Parallel (MP)",
        x_layout,
        expected_map=(-1, -1, -1, 0),
        extra_args=[2]
    )


def test_expanddims_hybrid_parallel_3():
    """
    Feature: Hybrid parallel.
    Description: insert dimension in middle, axis=1.
    Expectation: new dimension at position 1, others shifted.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "3. Hybrid Parallel (DP+CP+MP)",
        x_layout,
        expected_map=(2, -1, 1, 0),
        extra_args=[1]
    )


def test_expanddims_insert_at_end_4():
    """
    Feature: Insert at end.
    Description: insert dimension at the end, axis=-1.
    Expectation: new dimension appended at the end.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "4. Insert at end (axis=-1)",
        x_layout,
        expected_map=(2, 1, 0, -1),
        extra_args=[-1]
    )


def test_expanddims_negative_axis_5():
    """
    Feature: Negative axis indexing.
    Description: axis=-2 for rank-3 input, equivalent to axis=2.
    Expectation: new dimension at position 2.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "5. Negative axis (axis=-2)",
        x_layout,
        expected_map=(2, 1, -1, 0),
        extra_args=[-2]
    )


def test_expanddims_tuple_alias_dim_6():
    """
    Feature: Tuple alias in one dim.
    Description: insert at beginning with tuple alias ('dp','cp'), axis=0.
    Expectation: tuple alias preserved, new None at position 0.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "6. Tuple alias, insert at beginning",
        x_layout,
        expected_map=(-1, -1, (2, 1), 0),
        extra_args=[0]
    )


def test_expanddims_tuple_alias_dim_7():
    """
    Feature: Tuple alias in one dim.
    Description: insert in middle with tuple alias ('dp','cp'), axis=1.
    Expectation: new None inserted at position 1.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "7. Tuple alias, insert in middle",
        x_layout,
        expected_map=(-1, -1, (2, 1), 0),
        extra_args=[1]
    )


def test_expanddims_tuple_alias_dim_8():
    """
    Feature: Tuple alias in one dim.
    Description: insert at end with tuple alias, axis=2.
    Expectation: new None at position 2, mp shifted.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "8. Tuple alias, insert before mp",
        x_layout,
        expected_map=(-1, (2, 1), -1, 0),
        extra_args=[2]
    )


def test_expanddims_tuple_alias_dim_9():
    """
    Feature: Tuple alias in one dim.
    Description: insert before tuple alias, axis=0.
    Expectation: new None at position 0, tuple shifted to position 1.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout(("dp", "cp"), "mp")

    run_scenario(
        "9. Insert before tuple alias",
        x_layout,
        expected_map=(-1, (2, 1), 0),
        extra_args=[0]
    )


def test_expanddims_tuple_alias_dim_10():
    """
    Feature: Tuple alias in one dim.
    Description: insert after tuple alias, axis=1.
    Expectation: tuple at position 0, new None at position 1.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout(("dp", "cp"), "mp")

    run_scenario(
        "10. Insert after tuple alias",
        x_layout,
        expected_map=((2, 1), -1, 0),
        extra_args=[1]
    )


def test_expanddims_tuple_alias_with_none_11():
    """
    Feature: Tuple alias with None dimensions.
    Description: insert with tuple alias and None dims, axis=0.
    Expectation: new None at position 0.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        "11. Tuple alias with None dims",
        x_layout,
        expected_map=(-1, -1, (2, 1), -1),
        extra_args=[0]
    )


def test_expanddims_tuple_alias_negative_axis_12():
    """
    Feature: Tuple alias with negative axis.
    Description: insert with tuple alias, axis=-1.
    Expectation: new None at the end.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        "12. Tuple alias with negative axis",
        x_layout,
        expected_map=(-1, (2, 1), -1, -1),
        extra_args=[-1]
    )


def test_expanddims_scalar_to_1d_13():
    """
    Feature: Expand scalar.
    Description: expand scalar (rank-0) to 1D, axis=0.
    Expectation: output has one dimension with None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout()

    run_scenario(
        "13. Scalar to 1D",
        x_layout,
        expected_map=(-1,),
        extra_args=[0]
    )


def test_expanddims_extreme_negative_axis_14():
    """
    Feature: Extreme negative axis.
    Description: axis=-(rank+1), equivalent to axis=0.
    Expectation: insert at beginning.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "14. Extreme negative axis (axis=-4)",
        x_layout,
        expected_map=(-1, 2, 1, 0),
        extra_args=[-4]
    )


def test_expanddims_invalid_axis_15():
    """
    Feature: Invalid axis.
    Description: axis out of valid range.
    Expectation: raise ValueError.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    with pytest.raises(ValueError):
        run_scenario(
            "15. Invalid axis (axis=5)",
            x_layout,
            expected_map=(),
            extra_args=[5]
        )
