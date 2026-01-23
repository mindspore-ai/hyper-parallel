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
from hyper_parallel.core.shard.ops.parallel_elementwise import ElementWiseDistributedOp, AddDistributedOp


# Define all operators to test
op_name = "element_wise"
op = ElementWiseDistributedOp(op_name)

op_name_with_partial = "element_wise_with_partial"
op_with_partial = AddDistributedOp(op_name_with_partial)

base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "sp", "mp")
base_rank_list = list(range(8))
layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)


def run_scenario(scenario_name, x_layout, y_layout, expected_map, extra_args):
    """Infer layout of element-wise operator"""
    print(f"\n{'=' * 80}")
    print(f"Test {op_name}, Scenario: {scenario_name}")
    print("=" * 80)

    if x_layout.is_partial or y_layout.is_partial:
        op_ = op_with_partial
        op_name_ = op_name_with_partial
    else:
        op_ = op
        op_name_ = op_name
    output_layout = op_.infer_layout((x_layout, y_layout), extra_args)

    assert isinstance(output_layout, Layout), f"Expected output_layout to be Layout, got {type(output_layout)}"
    got_map = output_layout.to_dict()["tensor_map"]
    assert got_map == expected_map, (
        f"{op_name_} failed in scenario '{scenario_name}'. "
        f"Expected {expected_map}, got {got_map}"
    )


def run_single_input_scenario(scenario_name, x_layout, expected_map, extra_args):
    """Infer layout of element-wise operator with single input"""
    print(f"\n{'=' * 80}")
    print(f"Test {op_name}, Scenario: {scenario_name}")
    print("=" * 80)

    if x_layout.is_partial:
        op_ = op_with_partial
        op_name_ = op_name_with_partial
    else:
        op_ = op
        op_name_ = op_name
    output_layout = op_.infer_layout((x_layout,), extra_args)

    assert isinstance(output_layout, Layout), f"Expected output_layout to be Layout, got {type(output_layout)}"
    got_map = output_layout.to_dict()["tensor_map"]
    assert got_map == expected_map, (
        f"{op_name_} failed in scenario '{scenario_name}'. "
        f"Expected {expected_map}, got {got_map}"
    )


def test_single_input_partial_0():
    """
    Feature: Element-wise operator with single input
    Description: Single input with Partial status
    Expectation: Output keeps input layout with Partial
    """
    x_layout = layout("None", "None", "None")
    x_layout.set_partial_by_dev_axis("dp", "sum")

    with pytest.raises(ValueError) as exc_info:
        op.infer_layout((x_layout,), extra_args={"input_shapes": [(4, 8, 16)]})

    assert "has Partial status which is not allowed" in str(exc_info.value)


def test_single_input_replicate_1():
    """
    Feature: Element-wise operator with single input
    Description: Single input with all dimensions replicated
    Expectation: Output keeps input layout
    """
    x_layout = layout("None", "None", "None")

    run_single_input_scenario(
        "1. Single input - Replicate",
        x_layout,
        expected_map=(-1, -1, -1),
        extra_args={"input_shapes": [(4, 8, 16)]},
    )


def test_single_input_sharded_2():
    """
    Feature: Element-wise operator with single input
    Description: Single input with all dimensions sharded
    Expectation: Output keeps input layout
    """
    x_layout = layout("dp", "sp", "mp")

    run_single_input_scenario(
        "2. Single input - Sharded",
        x_layout,
        expected_map=(2, 1, 0),
        extra_args={"input_shapes": [(4, 8, 16)]},
    )


def test_single_input_partial_3():
    """
    Feature: Element-wise operator with single input
    Description: Single input with Partial status
    Expectation: Output keeps input layout with Partial
    """
    x_layout = layout("None", "None", "None")
    x_layout.set_partial_by_dev_axis("dp", "sum")

    output_layout = op_with_partial.infer_layout((x_layout,), extra_args={"input_shapes": [(4, 8, 16)]})

    assert isinstance(output_layout, Layout), f"Expected output_layout to be Layout, got {type(output_layout)}"
    assert output_layout.partial[layout.mesh.axis_index("dp")] == "sum"


def test_both_replicate_4():
    """
    Feature: Element-wise operator with two inputs
    Description: Both inputs replicated on all dimensions
    Expectation: Output replicated on all dimensions
    """
    x_layout = layout("None", "None", "None")
    y_layout = layout("None", "None", "None")

    run_scenario(
        "4. Both replicated",
        x_layout, y_layout,
        expected_map=(-1, -1, -1),
        extra_args={"input_shapes": [(4, 8, 16), (4, 8, 16)]},
    )


def test_both_sharded_same_5():
    """
    Feature: Element-wise operator with two inputs
    Description: Both inputs sharded on same dimensions
    Expectation: Output sharded on same dimensions
    """
    x_layout = layout("dp", "None", "mp")
    y_layout = layout("dp", "None", "mp")

    run_scenario(
        "5. Both sharded same",
        x_layout, y_layout,
        expected_map=(2, -1, 0),
        extra_args={"input_shapes": [(4, 8, 16), (4, 8, 16)]},
    )


def test_one_sharded_one_replicate_6():
    """
    Feature: Element-wise operator with two inputs
    Description: One input sharded, other replicated
    Expectation: Output sharded
    """
    x_layout = layout("dp", "None", "mp")
    y_layout = layout("None", "None", "None")

    run_scenario(
        "6. One sharded, one replicated",
        x_layout, y_layout,
        expected_map=(2, -1, 0),
        extra_args={"input_shapes": [(4, 8, 16), (4, 8, 16)]},
    )


def test_different_sharding_conflict_7():
    """
    Feature: Element-wise operator with two inputs
    Description: Different sharding patterns on same dimension
    Expectation: Raise ValueError (requires communication)
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("sp", "None", "None")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "should have same sharding pattern" in str(exc_info.value)


def test_sharded_broadcasts_to_replicated_8():
    """
    Feature: Element-wise operator with broadcasting
    Description: Sharded input broadcasts to replicated (size 1 -> N)
    Expectation: Raise ValueError (requires communication)
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")

    extra_args = {"input_shapes": [(1, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "Broadcasting dimension cannot be sharded" in str(exc_info.value)


def test_replicated_broadcasts_to_sharded_9():
    """
    Feature: Element-wise operator with broadcasting
    Description: Replicated input broadcasts to sharded (size 1 -> N)
    Expectation: Output sharded
    """
    x_layout = layout("None", "None", "None")
    y_layout = layout("dp", "None", "mp")

    run_scenario(
        "9. Replicated broadcasts to sharded",
        x_layout, y_layout,
        expected_map=(2, -1, 0),
        extra_args={"input_shapes": [(1, 8, 16), (4, 8, 16)]},
    )


def test_both_need_broadcast_10():
    """
    Feature: Element-wise operator with broadcasting
    Description: Both inputs need broadcast on different dimensions
    Expectation: Output merges sharding
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "sp", "None")

    run_scenario(
        "10. Both need broadcast on different dims",
        x_layout, y_layout,
        expected_map=(2, 1, -1),
        extra_args={"input_shapes": [(4, 1, 16), (1, 8, 16)]},
    )


def test_complex_broadcast_11():
    """
    Feature: Element-wise operator with broadcasting
    Description: Complex broadcast with multiple dimensions
    Expectation: Output merges sharding correctly
    """
    x_layout = layout("dp", "sp", "None")
    y_layout = layout("None", "None", "mp")

    run_scenario(
        "11. Complex broadcast",
        x_layout, y_layout,
        expected_map=(2, 1, 0),
        extra_args={"input_shapes": [(4, 8, 1), (1, 1, 16)]},
    )


def test_scalar_broadcast_12():
    """
    Feature: Element-wise operator with broadcasting
    Description: Scalar broadcasts to sharded tensor
    Expectation: Output sharded
    """
    x_layout = layout("None")
    y_layout = layout("dp", "sp", "mp")

    run_scenario(
        "12. Scalar broadcast to sharded",
        x_layout, y_layout,
        expected_map=(2, 1, 0),
        extra_args={"input_shapes": [(), (4, 8, 16)]},
    )


def test_partial_with_replicate_13():
    """
    Feature: Element-wise operator with Partial
    Description: Partial input with replicated input
    Expectation: Output Partial
    """
    x_layout = layout("None", "None", "None")
    x_layout.set_partial_by_dev_axis("dp", "sum")
    y_layout = layout("None", "None", "None")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}
    output_layout = op_with_partial.infer_layout((x_layout, y_layout), extra_args)

    assert isinstance(output_layout, Layout), f"Expected output_layout to be Layout, got {type(output_layout)}"
    assert output_layout.partial[layout.mesh.axis_index("dp")] == "sum"


def test_partial_with_partial_same_14():
    """
    Feature: Element-wise operator with Partial
    Description: Both inputs have same Partial operation
    Expectation: Output Partial with same operation
    """
    x_layout = layout("None", "None", "None")
    x_layout.set_partial_by_dev_axis("dp", "sum")
    y_layout = layout("None", "None", "None")
    y_layout.set_partial_by_dev_axis("dp", "sum")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}
    output_layout = op_with_partial.infer_layout((x_layout, y_layout), extra_args)

    assert isinstance(output_layout, Layout), f"Expected output_layout to be Layout, got {type(output_layout)}"
    assert output_layout.partial[layout.mesh.axis_index("dp")] == "sum"


def test_partial_with_partial_different_15():
    """
    Feature: Element-wise operator with Partial
    Description: Both inputs have different Partial operations
    Expectation: Raise ValueError
    """
    x_layout = layout("None", "None", "None")
    x_layout.set_partial_by_dev_axis("dp", "sum")
    y_layout = layout("None", "None", "None")
    y_layout.set_partial_by_dev_axis("dp", "max")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "partial operations should be same" in str(exc_info.value)


def test_shard_with_partial_conflict_16():
    """
    Feature: Element-wise operator with Shard and Partial
    Description: One input sharded, other Partial on same device axis
    Expectation: Raise ValueError
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")
    y_layout.set_partial_by_dev_axis("dp", "sum")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "Shard and Partial should not coexist on same device axis" in str(exc_info.value)


def test_partial_broadcasts_to_sharded_17():
    """
    Feature: Element-wise operator with Partial broadcasting
    Description: Partial input broadcasts to sharded input
    Expectation: Raise ValueError
    """
    x_layout = layout("None", "None", "None")
    x_layout.set_partial_by_dev_axis("dp", "sum")
    y_layout = layout("dp", "None", "None")

    extra_args = {"input_shapes": [(1, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "Shard and Partial should not coexist on same device axis" in str(exc_info.value)


def test_multiple_inputs_18():
    """
    Feature: Element-wise operator
    Description: 3 inputs
    Expectation: Run Successfully
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")
    z_layout = layout("None", "None", "None")

    extra_args = {"input_shapes": [(4, 8, 16), (1, 8, 16), (1, 8, 16)]}
    expected_map = (2, -1, -1)

    output_layout = op.infer_layout((x_layout, y_layout, z_layout), extra_args=extra_args)
    got_map = output_layout.to_dict()["tensor_map"]
    assert got_map == expected_map


def test_output_shard_partial_conflict_19():
    """
    Feature: Element-wise operator
    Description: Output would have both Shard and Partial on same axis
    Expectation: Raise ValueError
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")
    y_layout.set_partial_by_dev_axis("dp", "sum")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "Shard and Partial should not coexist on same device axis" in str(exc_info.value)


def test_incompatible_broadcast_shapes_20():
    """
    Feature: Element-wise operator with broadcasting
    Description: Incompatible shapes for broadcasting
    Expectation: Raise ValueError
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 7, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "cannot be broadcast" in str(exc_info.value)


def test_tuple_sharding_21():
    """
    Feature: Element-wise operator with tuple sharding
    Description: Inputs with tuple sharding (multiple axes sharding same dimension)
    Expectation: Output maintains tuple sharding when compatible
    """
    x_layout = layout(("dp", "sp"), "None", "None")
    y_layout = layout(("dp", "sp"), "None", "None")

    run_scenario(
        "21. Tuple sharding same",
        x_layout, y_layout,
        expected_map=((2, 1), -1, -1),
        extra_args={"input_shapes": [(4, 8, 16), (4, 8, 16)]},
    )


def test_different_tuple_sharding_22():
    """
    Feature: Element-wise operator with tuple sharding
    Description: Different tuple sharding patterns
    Expectation: Raise ValueError (requires communication)
    """
    x_layout = layout(("dp", "sp"), "None", "None")
    y_layout = layout(("dp", "mp"), "None", "None")

    extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

    with pytest.raises(ValueError) as exc_info:
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)

    assert "should have same sharding pattern" in str(exc_info.value)


def test_1d_to_3d_broadcast_23():
    """
    Feature: Element-wise operator with different dimension counts
    Description: 1D tensor broadcasts to 3D tensor
    Expectation: Output maintains 3D sharding
    """

    # 1D tensor: shape (16), sharded on mp
    x_layout = layout("mp")

    # 3D tensor: shape (4, 8, 16), sharded on dp and sp
    y_layout = layout("dp", "sp", "None")

    run_scenario(
        "23. 1D to 3D broadcast",
        x_layout, y_layout,
        expected_map=(2, 1, 0),  # dp在维度0，sp在维度1，mp在维度2
        extra_args={"input_shapes": [(16,), (4, 8, 16)]},
    )


def test_no_input_shapes_provided_24():
    """
    Feature: Element-wise operator
    Description: No input shapes provided
    Expectation: Raise ValueError
    """
    x_layout = layout("dp", "None", "None")
    y_layout = layout("None", "None", "None")

    with pytest.raises(ValueError) as exc_info:
        op.infer_layout((x_layout, y_layout), extra_args={})

    assert "cannot infer layout without shapes" in str(exc_info.value)
