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
"""parallel_reduce_max test"""

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_reduce import ReduceMaxDistributedOp


def run_scenario(scenario_name, x_layout, expected_map, extra_args):
    """Infer layout of reduce max operator"""
    print(f"\n{'=' * 80}")
    print(f"Test ReduceMax, Scenario: {scenario_name}")
    print('=' * 80)

    op = ReduceMaxDistributedOp("ReduceMax")
    output_layout = op.infer_layout((x_layout,), extra_args)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"ReduceMax failed in scenario '{scenario_name}'. " \
        f"Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_reduce_max_data_parallel_1():
    """
    Feature: Data parallel.
    Description: reduce dp axis, keepdim=False.
    Expectation: dp axis reduced.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")

    run_scenario(
        "1. Data Parallel (DP)",
        x_layout,
        expected_map=(-1, -1),
        extra_args=[0, False]
    )


def test_reduce_max_model_parallel_2():
    """
    Feature: Model parallel.
    Description: reduce mp axis, keepdim=True.
    Expectation: mp axis -> None.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "mp")

    run_scenario(
        "2. Model Parallel (MP)",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[2, True]
    )


def test_reduce_max_hybrid_parallel_3():
    """
    Feature: Hybrid parallel.
    Description: reduce cp axis, keepdim=False.
    Expectation: cp reduced, dp/mp kept.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "3. Hybrid Parallel (DP+CP+MP)",
        x_layout,
        expected_map=(2, 0),
        extra_args=[1, False]
    )


def test_reduce_max_reduce_multiple_dims_4():
    """
    Feature: Reduce over multiple dims.
    Description: reduce (0, 2), keepdim=True.
    Expectation: dp/mp -> None, cp kept.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "4. Reduce multiple dims (dp+mp)",
        x_layout,
        expected_map=(-1, 1, -1),
        extra_args=[(0, 2), True]
    )


def test_reduce_max_reduce_all_dims_5():
    """
    Feature: Reduce over all dims.
    Description: dim=None, keepdim=False.
    Expectation: all reduced.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "5. Reduce all dims",
        x_layout,
        expected_map=(),
        extra_args=[None, False]
    )


def test_reduce_max_tuple_alias_dim_6():
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=None with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "6. Tuple alias in one dim",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[None, True]
    )


def test_reduce_max_tuple_alias_dim_7():
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=None with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "7. Tuple alias in one dim",
        x_layout,
        expected_map=(),
        extra_args=[None, False]
    )


def test_reduce_max_tuple_alias_dim_8():
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 2) with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "8. Tuple alias in one dim",
        x_layout,
        expected_map=(-1, (2, 1), -1),
        extra_args=[(0, 2), True]
    )


def test_reduce_max_tuple_alias_dim_9():
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 2) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "9. Tuple alias in one dim",
        x_layout,
        expected_map=((2, 1),),
        extra_args=[(0, 2), False]
    )


def test_reduce_max_tuple_alias_dim_10():
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        "10. Tuple alias in one dim",
        x_layout,
        expected_map=(0,),
        extra_args=[(0, 1), False]
    )


def test_reduce_max_tuple_alias_dim_11():
    """
    Feature: Tuple alias in one dim, keep dim 'none'.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        "11. Tuple alias in one dim, with 'None', keepdim=True.",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[(0, 1), True]
    )


def test_reduce_max_tuple_alias_dim_12():
    """
    Feature: Tuple alias in one dim, keep other dim 'none'.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        "12. Tuple alias in one dim, with 'None', keepdim=False.",
        x_layout,
        expected_map=(-1,),
        extra_args=[(0, 1), False]
    )


def test_reduce_max_tuple_alias_dim_13():
    """
    Feature: Tuple alias in one dim, reduce with negative dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        "13. Tuple alias in one dim, with 'None', keepdim=True, reduce with negative dim.",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[(-3, -2), True]
    )


def test_reduce_max_tuple_alias_dim_14():
    """
    Feature: Tuple alias in one dim, reduce with negative dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        "14. Tuple alias in one dim, with 'None', keepdim=False, reduce with negative dim.",
        x_layout,
        expected_map=(-1,),
        extra_args=[(-3, -2), False]
    )


def test_reduce_max_tuple_alias_dim_15():
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: raise ValueError.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    with pytest.raises(ValueError):
        run_scenario(
            "15. Tuple alias in one dim, illegal reduce dim",
            x_layout,
            expected_map=(-1, -1),
            extra_args=[(-4, -2), False]
        )
