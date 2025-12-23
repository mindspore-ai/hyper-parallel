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
"""
Unit tests for element-wise distributed operators
"""

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_elementwise import (
    LessEqualDistributedOp,
    GreaterEqualDistributedOp,
    LogicalOrDistributedOp,
    MinimumDistributedOp,
)


# Define all operators to test
OPERATORS = [
    (LessEqualDistributedOp, "LessEqual"),
    (GreaterEqualDistributedOp, "GreaterEqual"),
    (LogicalOrDistributedOp, "LogicalOr"),
    (MinimumDistributedOp, "Minimum"),
]

base_device_matrix = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def run_scenario(operator_class, op_name, scenario_name, x_layout, y_layout, expected_map, extra_args):
    """Infer layout of element-wise operator"""
    print(f"\n{'=' * 80}")
    print(f"Test {op_name}, Scenario: {scenario_name}")
    print("=" * 80)

    op = operator_class(op_name)
    output_layout = op.infer_layout((x_layout, y_layout), extra_args)

    got_map = output_layout.to_dict()["tensor_map"]
    assert got_map == expected_map, (
        f"{op_name} failed in scenario '{scenario_name}'. "
        f"Expected {expected_map}, got {got_map}"
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_same_layout_dp_cp_mp_1(operator_class, op_name):
    """
    Feature: Element-wise operator.
    Description: Two inputs share the same layout (dp, cp, mp).
    Expectation: Output keeps the same tensor_map.
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "mp")

    run_scenario(
        operator_class, op_name,
        "1. Same layout (dp, cp, mp)",
        x_layout, y_layout,
        expected_map=(2, 1, 0),
        extra_args={},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_same_layout_dp_only_2(operator_class, op_name):
    """
    Feature: Element-wise operator.
    Description: Two inputs share layout (dp, None, None).
    Expectation: Output keeps dp sharding.
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "None")
    y_layout = layout("dp", "None", "None")

    run_scenario(
        operator_class, op_name,
        "2. Same layout (dp, None, None)",
        x_layout, y_layout,
        expected_map=(2, -1, -1),
        extra_args={},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_same_layout_mp_only_3(operator_class, op_name):
    """
    Feature: Element-wise operator.
    Description: Two inputs share layout (None, None, mp).
    Expectation: Output keeps mp sharding.
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("None", "None", "mp")
    y_layout = layout("None", "None", "mp")

    run_scenario(
        operator_class, op_name,
        "3. Same layout (None, None, mp)",
        x_layout, y_layout,
        expected_map=(-1, -1, 0),
        extra_args={},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_with_constant_rhs_4(operator_class, op_name):
    """
    Feature: Element-wise operator.
    Description: RHS is a scalar/constant (layout=None).
    Expectation: Output keeps LHS tensor layout.
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "cp", "mp")
    y_layout = None

    run_scenario(
        operator_class, op_name,
        "4. RHS is constant (layout=None)",
        x_layout, y_layout,
        expected_map=(2, 1, 0),
        extra_args={},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_mismatch_layout_should_raise_5(operator_class, op_name):
    """
    Feature: Element-wise operator.
    Description: Mismatched layouts without shape info are not allowed.
    Expectation: Raise ValueError.
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "None")

    op = operator_class(op_name)
    with pytest.raises(ValueError):
        op.infer_layout((x_layout, y_layout), extra_args={})


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_dimension_expansion_6(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (4, 8, 16) sharded on dp, y: (1, 8, 16) not sharded.
    Expectation: Output keeps x's layout (dp, None, None).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")

    run_scenario(
        operator_class, op_name,
        "6. Broadcast dim expansion (1 -> 4)",
        x_layout, y_layout,
        expected_map=(2, -1, -1),
        extra_args={"input_shapes": [(4, 8, 16), (1, 8, 16)]},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_different_dims_7(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (8, 16) sharded on (dp, mp), y: (16,) not sharded.
    Expectation: Output keeps x's layout (dp, mp).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "mp")
    y_layout = layout("None")

    run_scenario(
        operator_class, op_name,
        "7. Broadcast different dims (2D vs 1D)",
        x_layout, y_layout,
        expected_map=(2, 0),
        extra_args={"input_shapes": [(8, 16), (16,)]},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_merge_sharding_8(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (4, 1, 16) sharded on (dp, None, None), y: (1, 8, 16) sharded on (None, cp, None).
    Expectation: Output merges both sharding strategies: (dp, cp, None).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "cp", "None")

    run_scenario(
        operator_class, op_name,
        "8. Broadcast merge sharding from different axes",
        x_layout, y_layout,
        expected_map=(2, 1, -1),
        extra_args={"input_shapes": [(4, 1, 16), (1, 8, 16)]},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_3_inputs_9(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: Three inputs with different sharding on different dimensions.
    Expectation: Output merges all three: (dp, cp, mp).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "cp", "None")
    z_layout = layout("None", "None", "mp")

    op = operator_class(op_name)
    extra_args = {"input_shapes": [(4, 1, 1), (1, 8, 1), (1, 1, 16)]}
    out_layout = op.infer_layout((x_layout, y_layout, z_layout), extra_args=extra_args)

    got_map = out_layout.to_dict()["tensor_map"]
    expected_map = (2, 1, 0)
    assert got_map == expected_map, (
        f"{op_name} failed in 3-input broadcast scenario. "
        f"Expected {expected_map}, got {got_map}"
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_sharded_dim_error_10(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (1, 8, 16) with dimension 0 sharded on dp but size is 1.
    Expectation: Raise ValueError (broadcasting dimension cannot be sharded).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")

    op = operator_class(op_name)
    extra_args = {"input_shapes": [(1, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError, match="Broadcasting dimension cannot be sharded"):
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_conflicting_sharding_11(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (4, 8, 16) sharded on (dp, None, None), y: (4, 8, 16) sharded on (cp, None, None).
    Expectation: Raise ValueError (conflicting sharding).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "None")
    y_layout = layout("cp", "None", "None")

    op = operator_class(op_name)
    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError, match="Conflicting sharding"):
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_incompatible_shapes_12(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (4, 8, 16), y: (4, 7, 16) - dimension 1 has different non-unit sizes.
    Expectation: Raise ValueError (shapes cannot be broadcast).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "None")
    y_layout = layout("dp", "None", "None")

    op = operator_class(op_name)
    extra_args = {"input_shapes": [(4, 8, 16), (4, 7, 16)]}

    with pytest.raises(ValueError, match="cannot be broadcast"):
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_tuple_tensor_map_13(operator_class, op_name):
    """
    Feature: Element-wise operator with tuple tensor_map.
    Description: x: (4, 8, 16) sharded on ((dp, cp), None, None), y: (1, 8, 16) not sharded.
    Expectation: Output keeps x's tuple sharding (sorted as (1, 2)).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout(("dp", "cp"), "None", "None")
    y_layout = layout("None", "None", "None")

    run_scenario(
        operator_class, op_name,
        "13. Broadcast with tuple tensor_map",
        x_layout, y_layout,
        expected_map=((1, 2), -1, -1),
        extra_args={"input_shapes": [(4, 8, 16), (1, 8, 16)]},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_tuple_merge_14(operator_class, op_name):
    """
    Feature: Element-wise operator with tuple tensor_map.
    Description: x: (4, 1, 16) sharded on ((dp, cp), None, None), y: (1, 8, 16) sharded on (None, mp, None).
    Expectation: Output has merged sharding ((1, 2), 0, -1).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout(("dp", "cp"), "None", "None")
    y_layout = layout("None", "mp", "None")

    run_scenario(
        operator_class, op_name,
        "14. Broadcast merge tuple and single sharding",
        x_layout, y_layout,
        expected_map=((1, 2), 0, -1),
        extra_args={"input_shapes": [(4, 1, 16), (1, 8, 16)]},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_tuple_sharded_dim_error_15(operator_class, op_name):
    """
    Feature: Element-wise operator with tuple tensor_map.
    Description: x: (1, 8, 16) with dimension 0 sharded on (dp, cp) but size is 1.
    Expectation: Raise ValueError (broadcasting dimension cannot be sharded).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout(("dp", "cp"), "None", "None")
    y_layout = layout("None", "None", "None")

    op = operator_class(op_name)
    extra_args = {"input_shapes": [(1, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError, match="Broadcasting dimension cannot be sharded"):
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_tuple_conflict_16(operator_class, op_name):
    """
    Feature: Element-wise operator with tuple tensor_map.
    Description: x: (4, 8, 16) sharded on ((dp, cp), None, None), y: (4, 8, 16) sharded on ((dp, mp), None, None).
    Expectation: Raise ValueError (conflicting sharding).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout(("dp", "cp"), "None", "None")
    y_layout = layout(("dp", "mp"), "None", "None")

    op = operator_class(op_name)
    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError, match="Conflicting sharding"):
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_all_ones_17(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (1, 1, 1) not sharded, y: (4, 8, 16) sharded on (dp, cp, mp).
    Expectation: Output keeps y's layout (dp, cp, mp).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("None", "None", "None")
    y_layout = layout("dp", "cp", "mp")

    run_scenario(
        operator_class, op_name,
        "17. Broadcast from all-1 shape to full shape",
        x_layout, y_layout,
        expected_map=(2, 1, 0),
        extra_args={"input_shapes": [(1, 1, 1), (4, 8, 16)]},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_scalar_18(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (4, 8, 16) sharded on (dp, cp, mp), y is scalar (shape ()).
    Expectation: Output keeps x's layout (dp, cp, mp).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "cp", "mp")
    y_layout = None

    run_scenario(
        operator_class, op_name,
        "18. Broadcast scalar to tensor",
        x_layout, y_layout,
        expected_map=(2, 1, 0),
        extra_args={"input_shapes": [(4, 8, 16), ()]},
    )


@pytest.mark.parametrize("operator_class,op_name", OPERATORS)
def test_broadcast_both_need_expansion_19(operator_class, op_name):
    """
    Feature: Element-wise operator with broadcasting.
    Description: x: (4, 1, 16) sharded on (dp, None, mp), y: (1, 8, 1) sharded on (None, cp, None).
    Expectation: Output merges to (dp, cp, mp).
    """
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None", "mp")
    y_layout = layout("None", "cp", "None")

    run_scenario(
        operator_class, op_name,
        "19. Both inputs need dimension expansion",
        x_layout, y_layout,
        expected_map=(2, 1, 0),
        extra_args={"input_shapes": [(4, 1, 16), (1, 8, 1)]},
    )
