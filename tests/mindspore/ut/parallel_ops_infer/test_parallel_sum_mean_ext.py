# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_reduce import SumExtDistributedOp, MeanExtDistributedOp


def run_scenario(op_name, scenario_name, x_layout, expected_map, extra_args):
    """Infer layout of reduce operator"""
    print(f"\n{'=' * 80}")
    print(f"Test {op_name}, Scenario: {scenario_name}")
    print('=' * 80)

    op = SumExtDistributedOp(op_name) if op_name == "SumExt" else MeanExtDistributedOp(op_name)
    output_layout = op.infer_layout((x_layout,), extra_args)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"{op_name} failed in scenario '{scenario_name}'. " \
        f"Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


base_device_matrix = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_data_parallel_1(op_name):
    """
    Feature: Data parallel.
    Description: reduce dp axis, keepdim=False.
    Expectation: dp axis reduced.
    """
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")

    run_scenario(
        op_name,
        "1. Data Parallel (DP)",
        x_layout,
        expected_map=(-1, -1),
        extra_args=[0, False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_model_parallel_2(op_name):
    """
    Feature: Model parallel.
    Description: reduce mp axis, keepdim=True.
    Expectation: mp axis -> None.
    """
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "mp")

    run_scenario(
        op_name,
        "2. Model Parallel (MP)",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[2, True]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_hybrid_parallel_3(op_name):
    """
    Feature: Hybrid parallel.
    Description: reduce cp axis, keepdim=False.
    Expectation: cp reduced, dp/mp kept.
    """
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        op_name,
        "3. Hybrid Parallel (DP+CP+MP)",
        x_layout,
        expected_map=(2, 0),
        extra_args=[1, False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_reduce_multiple_dims_4(op_name):
    """
    Feature: Reduce over multiple dims.
    Description: reduce (0, 2), keepdim=True.
    Expectation: dp/mp -> None, cp kept.
    """
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        op_name,
        "4. Reduce multiple dims (dp+mp)",
        x_layout,
        expected_map=(-1, 1, -1),
        extra_args=[(0, 2), True]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_reduce_all_dims_5(op_name):
    """
    Feature: Reduce over all dims.
    Description: dim=None, keepdim=False.
    Expectation: all reduced.
    """
    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        op_name,
        "5. Reduce all dims",
        x_layout,
        expected_map=(),
        extra_args=[None, False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_6(op_name):
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=None with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        op_name,
        "6. Tuple alias in one dim",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[None, True]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_7(op_name):
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=None with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        op_name,
        "7. Tuple alias in one dim",
        x_layout,
        expected_map=(),
        extra_args=[None, False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_8(op_name):
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 2) with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        op_name,
        "8. Tuple alias in one dim",
        x_layout,
        expected_map=(-1, (2, 1), -1),
        extra_args=[(0, 2), True]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_9(op_name):
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 2) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        op_name,
        "9. Tuple alias in one dim",
        x_layout,
        expected_map=((2, 1),),
        extra_args=[(0, 2), False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_10(op_name):
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    run_scenario(
        op_name,
        "10. Tuple alias in one dim",
        x_layout,
        expected_map=(0,),
        extra_args=[(0, 1), False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_11(op_name):
    """
    Feature: Tuple alias in one dim, keep dim 'none'.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        op_name,
        "11. Tuple alias in one dim, with 'None', keepdim=True.",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[(0, 1), True]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_12(op_name):
    """
    Feature: Tuple alias in one dim, keep other dim 'none'.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        op_name,
        "12. Tuple alias in one dim, with 'None', keepdim=False.",
        x_layout,
        expected_map=(-1,),
        extra_args=[(0, 1), False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_13(op_name):
    """
    Feature: Tuple alias in one dim, reduce with negative dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=True.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        op_name,
        "13. Tuple alias in one dim, with 'None', keepdim=True, reduce with negative dim.",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args=[(-3, -2), True]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_14(op_name):
    """
    Feature: Tuple alias in one dim, reduce with negative dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: tuple replaced by None.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    run_scenario(
        op_name,
        "14. Tuple alias in one dim, with 'None', keepdim=False, reduce with negative dim.",
        x_layout,
        expected_map=(-1,),
        extra_args=[(-3, -2), False]
    )


@pytest.mark.parametrize("op_name", ["SumExt", "MeanExt"])
def test_sum_ext_tuple_alias_dim_15(op_name):
    """
    Feature: Tuple alias in one dim.
    Description: reduce dim=(0, 1) with alias ('dp','cp'), keepdim=False.
    Expectation: raise ValueError.
    """
    device_matrix = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(device_matrix, alias_name, rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "None")

    with pytest.raises(ValueError):
        run_scenario(
            op_name,
            "15. Tuple alias in one dim, illegal reduce dim",
            x_layout,
            expected_map=(-1, -1),
            extra_args=[(-4, -2), False]
        )
