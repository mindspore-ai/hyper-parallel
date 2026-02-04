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
"""parallel_one_hot_ext test"""

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_one_hot_ext import OneHotExtDistributedOp


def run_scenario(scenario_name, indices_layout, expected_map, extra_args):
    """Infer layout of OneHotExt operator"""
    print(f"\n{'=' * 80}")
    print(f"Test OneHotExt, Scenario: {scenario_name}")
    print("=" * 80)

    op = OneHotExtDistributedOp("OneHotExt")
    output_layout = op.infer_layout((indices_layout,), extra_args)
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"OneHotExt failed in scenario '{scenario_name}'. "
        f"Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def _base_layout():
    return Layout(base_mesh_shape, base_alias_name, base_rank_list)


def test_one_hot_ext_1d_axis_minus1_data_parallel_1():
    """
    Feature: 1D indices, axis=-1.
    Description: Indices sharded on dp.
    Expectation: Output keeps dp sharding and inserts replicated one-hot dim at end.
    """
    layout = _base_layout()
    indices_layout = layout("dp")

    run_scenario(
        "1. 1D + axis=-1 + DP sharding",
        indices_layout,
        expected_map=(2, -1),
        extra_args=[32, None, None, -1],
    )


def test_one_hot_ext_1d_axis_0_data_parallel_2():
    """
    Feature: 1D indices, axis=0.
    Description: Indices sharded on dp.
    Expectation: Insert replicated one-hot dim at position 0.
    """
    layout = _base_layout()
    indices_layout = layout("dp")

    run_scenario(
        "2. 1D + axis=0 + DP sharding",
        indices_layout,
        expected_map=(-1, 2),
        extra_args=[32, None, None, 0],
    )


def test_one_hot_ext_1d_axis_1_data_parallel_3():
    """
    Feature: 1D indices, axis=1.
    Description: axis=1 equals append for rank-1 tensor.
    Expectation: Same as axis=-1.
    """
    layout = _base_layout()
    indices_layout = layout("dp")

    run_scenario(
        "3. 1D + axis=1 (append) + DP sharding",
        indices_layout,
        expected_map=(2, -1),
        extra_args=[32, None, None, 1],
    )


def test_one_hot_ext_1d_replicate_all_4():
    """
    Feature: 1D indices fully replicated.
    Description: tensor_map is (-1,).
    Expectation: Output fully replicated with inserted one-hot dim replicated.
    """
    layout = _base_layout()
    indices_layout = layout("None")

    run_scenario(
        "4. 1D + replicate all",
        indices_layout,
        expected_map=(-1, -1),
        extra_args=[32, None, None, -1],
    )


def test_one_hot_ext_2d_axis_minus1_data_parallel_5():
    """
    Feature: 2D indices, axis=-1.
    Description: Only data-parallel (dim0 sharded), dim1 replicated.
    Expectation: Output keeps dp on dim0, inserts replicated one-hot dim at end.
    """
    layout = _base_layout()
    indices_layout = layout("dp", "None")

    run_scenario(
        "5. 2D + axis=-1 + DP only",
        indices_layout,
        expected_map=(2, -1, -1),
        extra_args=[64, None, None, -1],
    )


def test_one_hot_ext_3d_axis_minus1_data_parallel_6():
    """
    Feature: 3D indices, axis=-1.
    Description: Only data-parallel (dim0 sharded), others replicated.
    Expectation: Output keeps dp on dim0, inserts replicated one-hot dim at end.
    """
    layout = _base_layout()
    indices_layout = layout("dp", "None", "None")

    run_scenario(
        "6. 3D + axis=-1 + DP only",
        indices_layout,
        expected_map=(2, -1, -1, -1),
        extra_args=[16, None, None, -1],
    )


def test_one_hot_ext_2d_axis_not_minus1_should_raise_7():
    """
    Feature: Multi-dimensional restriction.
    Description: For rank>1, axis must be -1.
    Expectation: Raise ValueError.
    """
    layout = _base_layout()
    indices_layout = layout("dp", "None")

    op = OneHotExtDistributedOp("OneHotExt")
    with pytest.raises(ValueError):
        _ = op.infer_layout((indices_layout,), [64, None, None, 0])


def test_one_hot_ext_2d_not_data_parallel_should_raise_8():
    """
    Feature: Multi-dimensional restriction.
    Description: For rank>1, only dim0 can be sharded (DP only). Sharding dim1 is illegal.
    Expectation: Raise ValueError.
    """
    layout = _base_layout()
    indices_layout = layout("dp", "cp")

    op = OneHotExtDistributedOp("OneHotExt")
    with pytest.raises(ValueError):
        _ = op.infer_layout((indices_layout,), [64, None, None, -1])


def test_one_hot_ext_axis_out_of_range_should_raise_9():
    """
    Feature: Axis validation.
    Description: axis must be in [-1, 1] for this implementation.
    Expectation: Raise ValueError.
    """
    layout = _base_layout()
    indices_layout = layout("dp")

    op = OneHotExtDistributedOp("OneHotExt")
    with pytest.raises(ValueError):
        _ = op.infer_layout((indices_layout,), [32, None, None, 2])


def test_one_hot_ext_num_classes_invalid_should_raise_10():
    """
    Feature: num_classes validation.
    Description: num_classes must be int and >= -1.
    Expectation: Raise ValueError when num_classes < -1.
    """
    layout = _base_layout()
    indices_layout = layout("dp")

    op = OneHotExtDistributedOp("OneHotExt")
    with pytest.raises(ValueError):
        _ = op.infer_layout((indices_layout,), [-2, None, None, -1])
