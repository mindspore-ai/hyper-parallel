# Copyright 2026 Huawei Technologies Co., Ltd
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
"""parallel_squeeze test"""

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_squeeze import SqueezeDistributedOp


def run_scenario(scenario_name, x_layout, expected_map, axis, input_shape):
    """Infer layout of Squeeze operator"""
    print(f"\n{'=' * 80}")
    print(f"Test Squeeze, Scenario: {scenario_name}")
    print('=' * 80)

    # Prepare extra_args as list: [axis, input_shapes] if axis is not None
    # or just [input_shapes] if axis is None
    if axis is not None:
        extra_args = [axis, [input_shape]]
    else:
        extra_args = [[input_shape]]

    op = SqueezeDistributedOp("Squeeze")
    output_layout = op.infer_layout((x_layout,), extra_args)
    actual_map = output_layout.to_dict()["tensor_map"]
    print(f"Input shape: {input_shape}")
    print(f"Axis: {axis}")
    print(f"Input tensor_map: {x_layout.to_dict()['tensor_map']}")
    print(f"Expected tensor_map: {expected_map}")
    print(f"Actual tensor_map: {actual_map}")
    assert actual_map == expected_map, \
        f"Squeeze failed in scenario '{scenario_name}'. " \
        f"Expected {expected_map}, got {actual_map}"


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_squeeze_data_parallel_1():
    """
    Feature: Data parallel.
    Description: remove singleton dimension at position 1, axis=1.
    Expectation: singleton dimension removed, DP dimension at position 0 remains.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "None")
    input_shape = [8, 1, 4]  # position 1 has size 1

    run_scenario(
        "1. Data Parallel (DP) with singleton dimension",
        x_layout,
        expected_map=(2, -1),  # Remove position 1, leaving (2, -1)
        axis=1,
        input_shape=input_shape
    )


def test_squeeze_model_parallel_2():
    """
    Feature: Model parallel.
    Description: remove singleton dimension before mp axis, axis=0.
    Expectation: singleton dimension removed, MP dimension at position 0.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None", "mp")
    input_shape = [1, 1, 8]  # position 0 has size 1

    run_scenario(
        "2. Model Parallel (MP) with singleton dimensions",
        x_layout,
        expected_map=(-1, 0),  # Remove position 0 (shape=1), leaving (0,)
        axis=0,
        input_shape=input_shape
    )


def test_squeeze_hybrid_parallel_3():
    """
    Feature: Hybrid parallel.
    Description: remove singleton dimension in middle, axis=1.
    Expectation: singleton dimension removed, DP at position 0, CP at position 1, MP at position 2.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "cp", "mp")
    input_shape = [8, 1, 4, 2]  # position 1 has size 1

    run_scenario(
        "3. Hybrid Parallel (DP+CP+MP) with singleton dimension",
        x_layout,
        expected_map=(2, 1, 0),  # Remove position 1, leaving (2, 1, 0)
        axis=1,
        input_shape=input_shape
    )


def test_squeeze_remove_at_end_4():
    """
    Feature: Remove singleton at end.
    Description: remove singleton dimension at the end, axis=-1.
    Expectation: last singleton dimension removed.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp", "None")
    input_shape = [8, 4, 2, 1]  # last position has size 1

    run_scenario(
        "4. Remove singleton at end (axis=-1)",
        x_layout,
        expected_map=(2, 1, 0),  # Remove last position, leaving (2, 1, 0)
        axis=-1,
        input_shape=input_shape
    )


def test_squeeze_negative_axis_5():
    """
    Feature: Negative axis indexing.
    Description: axis=-2 for rank-3 input with singleton at position 1, equivalent to axis=1.
    Expectation: singleton dimension at position 1 removed.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "mp")
    input_shape = [8, 1, 2]  # position 1 has size 1

    run_scenario(
        "5. Negative axis (axis=-2)",
        x_layout,
        expected_map=(2, 0),  # Remove position 1, leaving (2, 0)
        axis=-2,
        input_shape=input_shape
    )


def test_squeeze_all_singletons_6():
    """
    Feature: Remove all singleton dimensions.
    Description: no axis specified, squeeze all singleton dimensions.
    Expectation: all shape=1 dimensions removed.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", "dp", "None", "cp", "None", "mp", "None")
    input_shape = [1, 8, 1, 4, 1, 2, 1]  # positions 0, 2, 4, 6 have size 1

    run_scenario(
        "6. Remove all singleton dimensions (axis=None)",
        x_layout,
        expected_map=(2, 1, 0),  # Remove all -1 at positions 0, 2, 4, 6, leaving (2, 1, 0)
        axis=None,
        input_shape=input_shape
    )


def test_squeeze_multiple_axes_7():
    """
    Feature: Remove multiple singleton dimensions.
    Description: specify multiple axes to squeeze.
    Expectation: specified singleton dimensions removed.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", "dp", "None", "cp", "None", "mp")
    input_shape = [1, 8, 1, 4, 1, 2]  # positions 0, 2, 4 have size 1

    run_scenario(
        "7. Remove multiple singleton dimensions",
        x_layout,
        expected_map=(2, 1, 0),  # Remove positions 0, 2, 4, leaving (2, 1, 0)
        axis=[0, 2, 4],
        input_shape=input_shape
    )


def test_squeeze_no_singletons_8():
    """
    Feature: No singleton dimensions to remove.
    Description: try to squeeze distributed dimensions.
    Expectation: ValueError should be raised.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "cp", "mp")
    input_shape = [8, 4, 2]  # No dimensions with size 1

    with pytest.raises(ValueError):
        run_scenario(
            "8. Try to squeeze non-singleton dimension",
            x_layout,
            expected_map=(),
            axis=1,
            input_shape=input_shape
        )


def test_squeeze_scalar_9():
    """
    Feature: Squeeze scalar.
    Description: squeeze all dimensions from scalar (rank-0).
    Expectation: scalar remains scalar (empty tensor map).
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout()
    input_shape = []  # scalar

    run_scenario(
        "9. Squeeze scalar (axis=None)",
        x_layout,
        expected_map=(),  # Empty tuple for scalar
        axis=None,
        input_shape=input_shape
    )


def test_squeeze_invalid_axis_10():
    """
    Feature: Invalid axis.
    Description: axis out of valid range.
    Expectation: raise ValueError.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "None", "mp")
    input_shape = [8, 1, 2]  # position 1 has size 1

    with pytest.raises(ValueError):
        run_scenario(
            "10. Invalid axis (axis=5)",
            x_layout,
            expected_map=(),
            axis=5,
            input_shape=input_shape
        )


def test_squeeze_negative_axes_list_11():
    """
    Feature: Multiple negative axes.
    Description: specify multiple negative axes to squeeze.
    Expectation: specified singleton dimensions removed.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", "dp", "None", "cp", "None", "mp")
    input_shape = [1, 8, 1, 4, 1, 2]  # positions 0, 2, 4 have size 1

    run_scenario(
        "11. Remove multiple singleton dimensions with negative axes",
        x_layout,
        expected_map=(2, 1, 0),  # Remove positions 0, 2, 4, leaving (2, 1, 0)
        axis=[-6, -4, -2],  # Positions 0, 2, 4
        input_shape=input_shape
    )


def test_squeeze_mixed_axes_12():
    """
    Feature: Mixed positive and negative axes.
    Description: specify mixed positive and negative axes.
    Expectation: specified singleton dimensions removed.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", "dp", "None", "cp", "None", "mp")
    input_shape = [1, 8, 1, 4, 1, 2]  # positions 0, 2, 4 have size 1

    run_scenario(
        "12. Remove singleton dimensions with mixed axes",
        x_layout,
        expected_map=(2, 1, 0),  # Remove positions 0, 2, 4, leaving (2, 1, 0)
        axis=[0, -4, 4],
        input_shape=input_shape
    )


def test_squeeze_partial_axes_13():
    """
    Feature: Partial axes removal.
    Description: squeeze only some singleton dimensions.
    Expectation: only specified singleton dimensions removed.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("None", "dp", "None", "cp", "None", "mp", "None")
    input_shape = [1, 8, 1, 4, 1, 2, 1]  # positions 0, 2, 4, 6 have size 1

    run_scenario(
        "13. Remove only some singleton dimensions",
        x_layout,
        expected_map=(2, 1, 0, -1),
        axis=[0, 2, 4],
        input_shape=input_shape
    )


def test_squeeze_shape_not_one_14():
    """
    Feature: Try to squeeze dimension with shape not equal to 1.
    Description: axis specified but shape is not 1.
    Expectation: ValueError should be raised.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "cp", "mp")
    input_shape = [8, 4, 2]  # No dimensions with size 1

    with pytest.raises(ValueError):
        run_scenario(
            "14. Try to squeeze dimension with shape not 1",
            x_layout,
            expected_map=(),
            axis=0,
            input_shape=input_shape
        )


def test_squeeze_distributed_dimension_15():
    """
    Feature: Try to squeeze distributed dimension.
    Description: axis specified for a dimension that is distributed.
    Expectation: ValueError should be raised.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "cp", "mp")
    input_shape = [8, 1, 2]  # position 1 has size 1 but is distributed

    with pytest.raises(ValueError):
        run_scenario(
            "15. Try to squeeze distributed dimension",
            x_layout,
            expected_map=(),
            axis=1,
            input_shape=input_shape
        )


def test_squeeze_no_input_shapes_16():
    """
    Feature: No input_shapes provided.
    Description: extra_args doesn't contain input_shapes.
    Expectation: ValueError should be raised.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    op = SqueezeDistributedOp("Squeeze")

    # Test with empty list (no input_shapes)
    with pytest.raises(ValueError):
        op.infer_layout((x_layout,), [])

    # Test with list containing only axis (no input_shapes)
    with pytest.raises(ValueError):
        op.infer_layout((x_layout,), [1])

    # Test with None
    with pytest.raises(ValueError):
        op.infer_layout((x_layout,), None)


def test_squeeze_scalar_with_axis_17():
    """
    Feature: Try to specify axis for scalar.
    Description: axis specified for scalar input.
    Expectation: ValueError should be raised.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))

    x_layout = Layout(mesh_shape, alias_name, rank_list)
    x_layout = x_layout()
    input_shape = []  # scalar

    op = SqueezeDistributedOp("Squeeze")

    # Test with axis specified for scalar
    with pytest.raises(ValueError):
        op.infer_layout((x_layout,), [0, [input_shape]])
