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
"""parallel_norm test"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_norm import NormDistributedOp
from hyper_parallel.core.shard.ops.parallel_norm import LayerNormDistributedOp

op = NormDistributedOp("RmsNorm")
torch_op = LayerNormDistributedOp("layer_naorm")


def test_rmsnorm_layout_data_parallel():
    """
    Feature: RmsNorm data parallel
    Description: Data parallel scenario with no splitting on normalization axis
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))

    # Data Parallel - input x with batch dimension sharded
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")  # 2D input: [batch, hidden]

    # Gamma layout (should be replicated)
    gamma_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    gamma_layout = gamma_layout("None", )  # 1D gamma: [hidden]

    # Beta layout (same as gamma)
    beta_layout = gamma_layout

    # Input layouts: x, gamma, beta
    input_layouts = (x_layout, gamma_layout, beta_layout)

    # Get output layout
    _, out_layout = op.infer_layout(input_layouts)

    # Expected output layout should match input x layout
    expected_map = (0, -1)  # Same as input x layout
    assert out_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel test failed. Expected {expected_map}, got {out_layout.to_dict()['tensor_map']}"


def test_rmsnorm_layout_tensor_parallel():
    """
    Feature: RmsNorm tensor parallel
    Description: Tensor parallel scenario with splitting on non-normalization axes
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # 3D input with sequence parallelism: [batch, seq, hidden]
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp", "None")  # Split batch and sequence dims

    # Gamma layout (should be replicated)
    gamma_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    gamma_layout = gamma_layout("None", )  # 1D gamma: [hidden]

    # Beta layout (same as gamma)
    beta_layout = None

    # Input layouts: x, gamma, beta
    input_layouts = (x_layout, gamma_layout, beta_layout)

    # Get output layout
    _, out_layout = op.infer_layout(input_layouts)

    # Expected output layout should match input x layout
    expected_map = (1, 0, -1)  # Same as input x layout
    assert out_layout.to_dict()["tensor_map"] == expected_map, \
        f"Tensor Parallel test failed. Expected {expected_map}, got {out_layout.to_dict()['tensor_map']}"


def test_rmsnorm_layout_mixed_parallel():
    """
    Feature: RmsNorm mixed parallel
    Description: Mixed data and tensor parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (4, 2)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # 3D input with both data and tensor parallelism: [batch, seq, hidden]
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout(("dp", "mp"), "None")  # Split batch and hidden dims

    # Gamma layout (should match the hidden dimension splitting)
    gamma_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    gamma_layout = gamma_layout("None", )  # 1D gamma: [hidden] with MP splitting

    # Beta layout (same as gamma)
    beta_layout = None

    # Input layouts: x, gamma, beta
    input_layouts = (x_layout, gamma_layout, beta_layout)

    # Get output layout
    _, out_layout = op.infer_layout(input_layouts)

    # Expected output layout should match input x layout
    expected_map = ((1, 0), -1)  # Same as input x layout
    assert out_layout.to_dict()["tensor_map"] == expected_map, \
        f"Mixed Parallel test failed. Expected {expected_map}, got {out_layout.to_dict()['tensor_map']}"


def test_rmsnorm_invalid_layout():
    """
    Feature: RmsNorm invalid layout
    Description: Test with invalid splitting on normalization axis
    Expectation: Raise ValueError
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Invalid: splitting on normalization axis (hidden dimension)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "mp")  # 2D input: [batch, hidden] with MP on hidden dim

    # Gamma layout (should match but can't due to normalization constraint)
    gamma_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    gamma_layout = gamma_layout("mp", )  # 1D gamma: [hidden] with MP splitting

    # Beta layout (same as gamma)
    beta_layout = gamma_layout

    # Input layouts: x, gamma, beta
    input_layouts = (x_layout, gamma_layout, beta_layout)

    # Should raise ValueError due to splitting on normalization axis
    try:
        _, _ = op.infer_layout(input_layouts)
        assert False, "Expected ValueError for invalid splitting on normalization axis"
    except ValueError as e:
        assert "RmsNorm is disabled to support the splitting after" in str(e)


def test_torch_layernorm_layout_data_parallel():
    """
    Feature: LayerNorm data parallel
    Description: Data parallel scenario with no splitting on normalization axis
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",)
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    out_layout = torch_op.infer_layout((x_layout,), extra_args=((64,),))

    expected_map = (0, -1)
    assert out_layout.to_dict()["tensor_map"] == expected_map


def test_torch_layernorm_layout_tensor_parallel():
    """
    Feature: LayerNorm tensor parallel
    Description: Tensor parallel scenario with splitting on non-normalization axes
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp", "None")

    # Normalize last dim
    out_layout = torch_op.infer_layout((x_layout,), extra_args=((128,),))

    expected_map = (1, 0, -1)
    assert out_layout.to_dict()["tensor_map"] == expected_map


def test_torch_layernorm_layout_normalized_dim_sharded():
    """
    Feature: Test error when normalized dimension is sharded
    Description: Test error when normalized dimension is sharded
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "mp")  # sharded on 'mp'

    with pytest.raises(ValueError, match="Cannot perform sharding on normalized dimension 1"):
        torch_op.infer_layout((x_layout,), extra_args=((64,),))


def test_torch_layernorm_layout_normalized_shape_too_large():
    """
    Feature: Test normalized_shape larger than input ndim
    Description: Test normalized_shape larger than input ndim
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "mp")
    with pytest.raises(ValueError, match="larger than input ndim"):
        torch_op.infer_layout((x_layout,), extra_args=((128, 64, 56),))
