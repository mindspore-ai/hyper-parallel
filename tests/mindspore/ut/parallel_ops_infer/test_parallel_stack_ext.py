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
Unit tests for StackExt distributed operators
"""
import pytest

from hyper_parallel import Layout
from tests.custom_ops.parallel_stack_ext import StackExtDistributedOp

base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def _infer_layout(op, layouts, axis):
    # StackExt: extra_args = [axis]
    return op.infer_layout(tuple(layouts), extra_args=[axis])


def run_scenario(scenario_name, layouts, axis, expected_map):
    print(f"\n{'=' * 80}")
    print(f"Test StackExt, Scenario: {scenario_name}")
    print("=" * 80)

    op = StackExtDistributedOp("StackExt")
    out_layout = _infer_layout(op, layouts, axis)

    got_map = out_layout.to_dict()["tensor_map"]
    assert got_map == expected_map, (
        f"StackExt failed in scenario '{scenario_name}'. "
        f"Expected {expected_map}, got {got_map}"
    )


def test_stack_ext_dispatch_and_layout():
    """
    Feature: StackExt distributed op dispatch and layout inference
    Description: Get StackExtDistributedOp via direct import (preferred) or registry (fallback),
                 then infer output layout for two inputs with identical layout using axis=0.
    Expectation: infer_layout succeeds and output tensor_map equals (-1, 2, 1, 0).
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = layout("dp", "cp", "mp")

    op = StackExtDistributedOp("StackExt")
    out_layout = _infer_layout(op, [x1, x2], axis=0)
    assert out_layout.to_dict()["tensor_map"] == (-1, 2, 1, 0)


def test_stack_ext_layout_cache():
    """
    Feature: StackExt layout inference stability (cache-like behavior)
    Description: Call infer_layout twice with the same inputs and axis to verify repeated calls
                 produce consistent layout results.
    Expectation: Both infer_layout calls succeed and produce identical tensor_map results.
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = layout("dp", "cp", "mp")

    op = StackExtDistributedOp("StackExt")
    out1 = _infer_layout(op, [x1, x2], axis=1)
    out2 = _infer_layout(op, [x1, x2], axis=1)

    assert out1.to_dict()["tensor_map"] == out2.to_dict()["tensor_map"]


def test_stack_ext_axis0_same_layout_1():
    """
    Feature: StackExt output layout inference with inserted dimension
    Description: Inputs share the same layout (dp, cp, mp) and use axis=0 to insert a new
                 leading dimension on the output layout.
    Expectation: infer_layout succeeds and output tensor_map equals (-1, 2, 1, 0).
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = layout("dp", "cp", "mp")

    run_scenario(
        "1. Same layout (dp, cp, mp), axis=0",
        layouts=[x1, x2],
        axis=0,
        expected_map=(-1, 2, 1, 0),
    )


def test_stack_ext_axis1_same_layout_2():
    """
    Feature: StackExt output layout inference with inserted dimension
    Description: Inputs share the same layout (dp, cp, mp) and use axis=1 to insert a new
                 dimension after dp on the output layout.
    Expectation: infer_layout succeeds and output tensor_map equals (2, -1, 1, 0).
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = layout("dp", "cp", "mp")

    run_scenario(
        "2. Same layout, axis=1",
        layouts=[x1, x2],
        axis=1,
        expected_map=(2, -1, 1, 0),
    )


def test_stack_ext_axis_minus1_3():
    """
    Feature: StackExt output layout inference with inserted dimension
    Description: Inputs share the same layout (dp, cp, mp) and use axis=-1 to append a new
                 trailing dimension on the output layout.
    Expectation: infer_layout succeeds and output tensor_map equals (2, 1, 0, -1).
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = layout("dp", "cp", "mp")

    run_scenario(
        "3. Same layout, axis=-1",
        layouts=[x1, x2],
        axis=-1,
        expected_map=(2, 1, 0, -1),
    )


def test_stack_ext_with_none_layout_4():
    """
    Feature: StackExt layout inference with optional None-layout inputs
    Description: Provide one normal layout input and one None layout input (constant-like).
                 The None layout input should be ignored in layout consistency checks.
    Expectation: infer_layout succeeds and output tensor_map follows the base layout insertion,
                 resulting in (-1, 2, 1, 0) for axis=0.
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = None

    run_scenario(
        "4. One input is None layout",
        layouts=[x1, x2],
        axis=0,
        expected_map=(-1, 2, 1, 0),
    )


def test_stack_ext_mismatch_tensor_map_should_raise_5():
    """
    Feature: StackExt input layout validation
    Description: Provide two non-None inputs with the same mesh_shape but different tensor_map
                 (e.g., one uses mp while the other uses None on the last dimension).
    Expectation: infer_layout raises ValueError due to tensor_map mismatch among non-None inputs.
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = layout("dp", "cp", "None")  # tensor_map differs: (2,1,-1) vs (2,1,0)

    op = StackExtDistributedOp("StackExt")
    with pytest.raises(ValueError):
        _ = _infer_layout(op, [x1, x2], axis=0)


def test_stack_ext_mismatch_mesh_shape_should_raise_6():
    """
    Feature: StackExt input layout validation
    Description: Provide two non-None inputs with different mesh_shape shapes.
                 Note: rank_list length must match prod(mesh_shape) for each layout instance.
    Expectation: infer_layout raises ValueError due to mesh_shape mismatch among inputs.
    """
    layout1 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout1("dp", "cp", "mp")

    layout2 = Layout((2, 2, 1), base_alias_name, list(range(4)))
    x2 = layout2("dp", "cp", "mp")

    op = StackExtDistributedOp("StackExt")
    with pytest.raises(ValueError):
        _ = _infer_layout(op, [x1, x2], axis=0)


def test_stack_ext_axis_out_of_range_should_raise_7():
    """
    Feature: StackExt axis range validation
    Description: Input rank=3 so output rank=4; axis must be within [-4, 3].
                 Provide an invalid axis=4 to trigger out-of-range validation.
    Expectation: infer_layout raises ValueError because axis is out of the valid range.
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x1 = layout("dp", "cp", "mp")
    x2 = layout("dp", "cp", "mp")

    op = StackExtDistributedOp("StackExt")
    with pytest.raises(ValueError):
        _ = _infer_layout(op, [x1, x2], axis=4)
        