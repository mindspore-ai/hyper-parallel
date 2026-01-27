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
"""parallel_transpose_ext_view test"""

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_transpose import TransposeDistributedOp


def run_scenario(scenario_name, x_layout, expected_map, extra_args):
    """Infer layout of TransposeExtView operator and validate tensor_map."""
    print(f"\n{'=' * 80}")
    print(f"Test TransposeExtView, Scenario: {scenario_name}")
    print("=" * 80)

    op = TransposeDistributedOp("TransposeExtView")
    output_layout = op.infer_layout((x_layout,), extra_args)
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"TransposeExtView failed in scenario '{scenario_name}'. "
        f"Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_transpose_ext_view_basic_swap_3d_1():
    """
    Feature: Basic swap.
    Description: swap dim0=0 and dim1=2 on 3D tensor map.
    Expectation: tensor_map dims swapped.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "1. Basic swap (0 <-> 2)", x_layout, expected_map=(0, 1, 2), extra_args=(0, 2)
    )


def test_transpose_ext_view_negative_dims_2():
    """
    Feature: Negative dims.
    Description: swap dim0=-1 and dim1=-3 on 3D tensor map.
    Expectation: normalized dims swapped.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    # ndim=3: -1 -> 2, -3 -> 0 => swap(2,0)
    run_scenario(
        "2. Negative dims (-1 <-> -3)",
        x_layout,
        expected_map=(0, 1, 2),
        extra_args=(-1, -3),
    )


def test_transpose_ext_view_noop_same_dims_3():
    """
    Feature: No-op.
    Description: dim0 == dim1.
    Expectation: output tensor_map unchanged.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    run_scenario(
        "3. No-op swap (1 <-> 1)", x_layout, expected_map=(2, 1, 0), extra_args=(1, 1)
    )


def test_transpose_ext_view_tuple_alias_dim_4():
    """
    Feature: Tuple alias dim.
    Description: swap a normal dim with a tuple-alias dim.
    Expectation: tuple moved to the swapped position.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", ("dp", "cp"), "mp")

    # tensor_map should be (-1, (2,1), 0) with base_alias_name ("dp","cp","mp")
    # swap dim0=1, dim1=2 => (-1, 0, (2,1))
    run_scenario(
        "4. Tuple alias swap (1 <-> 2)",
        x_layout,
        expected_map=(-1, 0, (2, 1)),
        extra_args=(1, 2),
    )


def test_transpose_ext_view_dim_out_of_range_5():
    """
    Feature: Error handling.
    Description: dim0 or dim1 out of range [-ndim, ndim-1].
    Expectation: raise ValueError.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    with pytest.raises(ValueError):
        run_scenario(
            "5. dim0 out of range", x_layout, expected_map=(2, 1, 0), extra_args=(3, 0)
        )

    with pytest.raises(ValueError):
        run_scenario(
            "6. dim1 out of range (negative)",
            x_layout,
            expected_map=(2, 1, 0),
            extra_args=(-4, 0),
        )


def test_transpose_ext_view_dim_type_error_6():
    """
    Feature: Error handling.
    Description: dim0 or dim1 is not int.
    Expectation: raise ValueError.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    with pytest.raises(ValueError):
        run_scenario(
            "7. dim0 type error", x_layout, expected_map=(2, 1, 0), extra_args=("0", 1)
        )

    with pytest.raises(ValueError):
        run_scenario(
            "8. dim1 type error", x_layout, expected_map=(2, 1, 0), extra_args=(0, None)
        )


def test_transpose_ext_view_extra_args_invalid_7():
    """
    Feature: Error handling.
    Description: extra_args is not (dim0, dim1).
    Expectation: raise ValueError.
    """
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "cp", "mp")

    with pytest.raises(ValueError):
        run_scenario(
            "9. extra_args missing dim1",
            x_layout,
            expected_map=(2, 1, 0),
            extra_args=(0,),
        )

    with pytest.raises((ValueError, TypeError)):
        run_scenario(
            "10. extra_args not tuple/list",
            x_layout,
            expected_map=(2, 1, 0),
            extra_args=None,
        )
