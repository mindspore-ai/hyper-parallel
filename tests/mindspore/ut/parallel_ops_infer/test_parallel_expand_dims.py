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

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_expand_dims import ExpandDimsDistributedOp


def run_scenario(scenario_name, x_layout, expected_map, extra_args):
    """Infer layout of ExpandDims operator"""
    print(f"\n{'=' * 80}")
    print(f"Test ExpandDims, Scenario: {scenario_name}")
    print('=' * 80)

    op = ExpandDimsDistributedOp("ExpandDims")
    output_layout = op.infer_layout((x_layout,), extra_args)
    assert output_layout.tensor_map == expected_map, (
        f"ExpandDims failed in scenario '{scenario_name}'. "
        f"Expected {expected_map}, got {output_layout.tensor_map}"
    )
    assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
        f"ExpandDims get_expand_impl failed in scenario '{scenario_name}'. "
        f"Expected None, got {op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
    )


def _build_mesh():
    """Create device mesh for testing."""
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "cp", "mp"),
        init_backend=False
    )


def test_expanddims_data_parallel_1(mesh):
    """
    Feature: Data parallel.
    Description: insert dimension at beginning, axis=0.
    Expectation: new dimension inserted at position 0, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    run_scenario(
        "1. Data Parallel (DP)",
        x_layout,
        expected_map=(-1, 2, -1, -1),
        extra_args=[0]
    )


def test_expanddims_model_parallel_2(mesh):
    """
    Feature: Model parallel.
    Description: insert dimension before mp axis, axis=2.
    Expectation: new dimension at position 2, mp shifted to position 3, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Replicate(), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    run_scenario(
        "2. Model Parallel (MP)",
        x_layout,
        expected_map=(-1, -1, -1, 0),
        extra_args=[2]
    )


def test_expanddims_hybrid_parallel_3(mesh):
    """
    Feature: Hybrid parallel.
    Description: insert dimension in middle, axis=1.
    Expectation: new dimension at position 1, others shifted, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Shard(0), Shard(1), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    run_scenario(
        "3. Hybrid Parallel (DP+CP+MP)",
        x_layout,
        expected_map=(2, -1, 1, 0),
        extra_args=[1]
    )


def test_expanddims_insert_at_end_4(mesh):
    """
    Feature: Insert at end.
    Description: insert dimension at the end, axis=-1.
    Expectation: new dimension appended at the end, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Shard(0), Shard(1), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    run_scenario(
        "4. Insert at end (axis=-1)",
        x_layout,
        expected_map=(2, 1, 0, -1),
        extra_args=[-1]
    )


def test_expanddims_negative_axis_5(mesh):
    """
    Feature: Negative axis indexing.
    Description: axis=-2 for rank-3 input, equivalent to axis=2.
    Expectation: new dimension at position 2, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Shard(0), Shard(1), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    run_scenario(
        "5. Negative axis (axis=-2)",
        x_layout,
        expected_map=(2, 1, -1, 0),
        extra_args=[-2]
    )


def test_expanddims_all_replicated_6():
    """
    Feature: All replicated.
    Description: insert dimension with all replicated input.
    Expectation: new dimension inserted with None, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Replicate(), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    run_scenario(
        "6. All Replicated",
        x_layout,
        expected_map=(-1, -1, -1, -1),
        extra_args=[1]
    )


def test_expanddims_2d_tensor_7():
    """
    Feature: 2D tensor.
    Description: insert dimension in 2D tensor.
    Expectation: correct tensor_map for 3D output, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Shard(0), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    run_scenario(
        "7. 2D Tensor",
        x_layout,
        expected_map=(2, -1, 1),
        extra_args=[1]
    )


def test_expanddims_scalar_to_1d_8():
    """
    Feature: Expand scalar.
    Description: expand scalar (rank-0) to 1D, axis=0.
    Expectation: output has one dimension with None, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Replicate(), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 0)

    run_scenario(
        "8. Scalar to 1D",
        x_layout,
        expected_map=(-1,),
        extra_args=[0]
    )


def test_expanddims_extreme_negative_axis_9():
    """
    Feature: Extreme negative axis.
    Description: axis=-(rank+1), equivalent to axis=0.
    Expectation: insert at beginning, get_expand_impl returns None.
    """
    mesh = _build_mesh()
    x_placements = (Shard(0), Shard(1), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    run_scenario(
        "9. Extreme negative axis (axis=-4)",
        x_layout,
        expected_map=(-1, 2, 1, 0),
        extra_args=[-4]
    )


def test_expanddims_invalid_axis_10():
    """
    Feature: Invalid axis.
    Description: axis out of valid range.
    Expectation: raise ValueError.
    """
    mesh = _build_mesh()
    x_placements = (Shard(0), Shard(1), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    with pytest.raises(ValueError, match="out of range for input rank"):
        run_scenario(
            "10. Invalid axis (axis=5)",
            x_layout,
            expected_map=(),
            extra_args=[5]
        )
