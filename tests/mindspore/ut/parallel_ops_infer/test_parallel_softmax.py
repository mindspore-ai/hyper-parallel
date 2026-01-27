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
"""parallel_softmax test"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_activation_with_axis import ActivationWithAxisDistributedOp


op = ActivationWithAxisDistributedOp("Softmax")


def test_softmax_layout_data_parallel():
    """
    Feature: Softmax data parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    output_layout = op.infer_layout((x_layout,), (-1,))
    expected_map = (0, -1)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Data Parallel with transpose_a test failed. Expected {expected_map},"
        f" got {output_layout.to_dict()['tensor_map']}"
    )


def test_softmax_layout_data_parallel_value_failed():
    """
    Feature: Softmax data parallel value failed
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    with pytest.raises(ValueError):
        _ = op.infer_layout((x_layout,), (0,))

op = ActivationWithAxisDistributedOp("softmax")


def test_torch_softmax_data_parallel_success():
    """
    Feature: Softmax compatible with PyTorch (dim arg)
    Description: Standard Data Parallel scenario (Batch is sharded, Softmax on Feature dim)
    Expectation: Success, output layout equals input layout
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    output_layout = op.infer_layout((x_layout,), (1,))

    expected_map = x_layout.to_dict()["tensor_map"]
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"DP test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


def test_torch_softmax_negative_dim_success():
    """
    Feature: Softmax compatible with PyTorch
    Description: Test negative index support (dim=-1) in Data Parallel
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # [Batch, Hidden] -> ("dp", "None")
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    output_layout = op.infer_layout((x_layout,), (-1,))

    expected_map = x_layout.to_dict()["tensor_map"]
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_torch_softmax_sharded_dim_failure():
    """
    Feature: Softmax distributed check
    Description: Attempting to compute Softmax on a sharded dimension
    Expectation: Raise ValueError
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # [Batch, Hidden] -> ("dp", "None")
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    with pytest.raises(ValueError) as excinfo:
        _ = op.infer_layout((x_layout,), (0,))

    assert "is sharded" in str(excinfo.value)
    assert "requires the reduction axis to be un-sharded" in str(excinfo.value)


def test_torch_softmax_model_parallel_failure():
    """
    Feature: Softmax distributed check
    Description: Model Parallel scenario where the feature dimension is sharded
    Expectation: Raise ValueError when computing softmax on the MP axis
    """
    base_mesh_shape = (8,)
    base_alias_name = ("mp",)
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "mp")

    with pytest.raises(ValueError) as excinfo:
        _ = op.infer_layout((x_layout,), (-1,))

    assert "is sharded" in str(excinfo.value)


def test_torch_softmax_model_parallel_success_on_other_axis():
    """
    Feature: Softmax distributed check
    Description: Model Parallel scenario, but computing softmax on the non-sharded axis
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("mp",)
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "mp")

    output_layout = op.infer_layout((x_layout,), (0,))

    expected_map = x_layout.to_dict()["tensor_map"]
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_torch_softmax_input_consistency_failure():
    """
    Feature: Layout consistency check
    Description: Inputs have different layouts
    Expectation: Raise ValueError
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    layout1 = Layout(base_mesh_shape, base_alias_name, base_rank_list)("dp", "None")
    layout2 = Layout(base_mesh_shape, base_alias_name, base_rank_list)("None", "dp")

    with pytest.raises(ValueError) as excinfo:
        _ = op.infer_layout((layout1, layout2), (1,))

    assert "requires all tensor inputs to have the same layout" in str(excinfo.value)


def test_torch_softmax_multi_axis_mesh_success():
    """
    Feature: Softmax with multi-axis device mesh
    Description: 2D mesh (dp+mp), Softmax applied to unsharded dimension
    Expectation: Success, output layout matches input
    """

    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None", "mp")

    output_layout = op.infer_layout((x_layout,), (1,))

    expected_map = x_layout.to_dict()["tensor_map"]
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Multi-axis mesh test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


def test_torch_softmax_interleaved_parallel_success():
    """
    Feature: Softmax with interleaved parallel
    Description: Softmax applied to unsharded dimension under interleaved parallel layout
    Expectation: Success
    """
    base_mesh_shape = (2, 2)
    base_alias_name = ("dp", "interleaved_parallel")
    base_rank_list = list(range(4))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    output_layout = op.infer_layout((x_layout,), (1,))

    assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]
    assert output_layout.to_dict()["interleaved_parallel"] is True

def test_torch_softmax_partial_layout_success():
    """
    Feature: Softmax with partial (reduce) layout
    Description: Softmax applied to unreduced/unsharded dimension under Partial layout
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "mp")  # tensor_map=(-1, 0)
    # Set partial sum (reduction) on dp axis
    x_layout.set_partial_by_dev_axis("dp", "sum")

    # Softmax applied to Batch dimension (dim=0) -> tensor_map=-1 (unsharded, only partial), valid
    output_layout = op.infer_layout((x_layout,), (0,))

    assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]
    assert output_layout.partial == x_layout.partial  # Partial status preserved


def test_torch_softmax_non_contiguous_rank_list():
    """
    Feature: Softmax with non-contiguous rank list
    Description: Non-contiguous rank_list (simulating heterogeneous cluster), verify layout compatibility
    Expectation: Success if dim is unsharded
    """
    base_mesh_shape = (2, 2)
    base_alias_name = ("dp", "mp")
    base_rank_list = [3, 1, 0, 2]  # Non-contiguous rank_list

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "mp")  # tensor_map=(-1, 0)

    # Softmax applied to Batch dimension (dim=0) -> unsharded, valid
    output_layout = op.infer_layout((x_layout,), (0,))

    assert output_layout.rank_list == tuple(base_rank_list)  # rank_list preserved
    assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]

def test_torch_softmax_nested_tensor_map_success():
    """
    Feature: Softmax with nested tensor_map (interleaved parallel tuple)
    Description: tensor_map contains nested tuple (interleaved parallel multi-axis binding)
    Expectation: Success if target dim is unsharded
    """
    base_mesh_shape = (2, 2, 2)
    base_alias_name = ("dp", "sp", "interleaved_parallel")
    base_rank_list = list(range(8))

    # 3D Tensor -> (("dp", "interleaved_parallel"), "None", "sp")
    # tensor_map calculation:
    # ("dp", "interleaved_parallel") -> (2, 0) (dp: 3-1-0=2; interleaved_parallel: 3-1-2=0)
    # "None" -> -1
    # "sp" -> 3-1-1=1
    # Final tensor_map = ((2, 0), -1, 1)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout(("dp", "interleaved_parallel"), "None", "sp")

    # Softmax applied to SeqLen dimension (dim=1) -> -1 (unsharded), valid
    output_layout = op.infer_layout((x_layout,), (1,))

    assert output_layout.to_dict()["tensor_map"] == ((2, 0), -1, 1)
    assert output_layout.to_dict()["interleaved_parallel"] is True

def test_torch_softmax_multi_input_same_layout_success():
    """
    Feature: Softmax with multiple inputs (same layout)
    Description: Multiple inputs with consistent layout, verify validity
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # Two input layouts are exactly the same
    layout1 = Layout(base_mesh_shape, base_alias_name, base_rank_list)("dp", "None")
    layout2 = Layout(base_mesh_shape, base_alias_name, base_rank_list)("dp", "None")

    # Softmax applied to unsharded dimension (dim=1)
    output_layout = op.infer_layout((layout1, layout2), (1,))

    assert output_layout.to_dict()["tensor_map"] == layout1.to_dict()["tensor_map"]
