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
"""parallel_embedding test"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_embedding import EmbeddingDistributedOp

op = EmbeddingDistributedOp("Embedding")
torch_op = EmbeddingDistributedOp("embedding")

def test_embedding_layout_parallel_1():
    """
    Feature: Embedding parallel
    Description: Parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None")

    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("a", "b")

    with pytest.raises(ValueError):
        output_layout = op.infer_layout(
            (
                x_layout,
                w_layout,
                None,
                None,
                None,
                None,
            ),
            (
                None,
                None,
            ),
        )
        expected_map = (1, -1, 0)  # Expected output tensor map
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )


def test_embedding_layout_parallel_2():
    """
    Feature: Embedding parallel
    Description: Parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (1, 8)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None")

    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("None", "b")

    output_layout = op.infer_layout(
        (
            x_layout,
            w_layout,
            None,
            None,
            None,
            None,
        ),
        (
            None,
            None,
        ),
    )
    expected_map = (-1, -1, 0)  # Expected output tensor map
    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


def test_torch_embedding_layout_dp_mp():
    """
    Feature: PyTorch Embedding parallel
    Description: Hybrid Parallel (Data Parallel on Input + Model Parallel on Weight Column)
    Logic:
        Input [Batch, Seq] -> Layout ("dp", "None") -> Map (1, -1)
        Weight [Vocab, Embed] -> Layout ("None", "mp") -> Map (-1, 0)
        Output [Batch, Seq, Embed] -> Should combine input map and weight last dim map
        Expected Map -> (1, -1, 0) corresponding to ("dp", "None", "mp")
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp") # dp index=0(val=1), mp index=1(val=0)
    base_rank_list = list(range(8))

    # Input: Sharded on Batch dimension (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    # Weight: Sharded on Embedding dimension (Column Parallel / MP)
    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("None", "mp")

    # infer_layout arguments: (layouts, extra_args)
    output_layout = torch_op.infer_layout(
        [x_layout, w_layout],
        None
    )

    # Output should inherit Batch sharding from Input and Embed sharding from Weight
    # "dp" is index 0 in aliases -> tensor_map value: len(2) - 1 - 0 = 1
    # "mp" is index 1 in aliases -> tensor_map value: len(2) - 1 - 1 = 0
    expected_map = (1, -1, 0)

    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


def test_torch_embedding_layout_pure_dp():
    """
    Feature: PyTorch Embedding parallel
    Description: Pure Data Parallel
    Logic:
        Input [Batch, Seq] -> Layout ("dp", "None") -> Map (0, -1) [Assuming 1D mesh]
        Weight [Vocab, Embed] -> Layout ("None", "None") -> Map (-1, -1)
        Output [Batch, Seq, Embed] -> Expected ("dp", "None", "None")
    Expectation: Success
    """
    base_mesh_shape = (8,)
    base_alias_name = ("dp",) # dp index=0 (val=0)
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("None", "None")

    output_layout = torch_op.infer_layout(
        [x_layout, w_layout],
        None
    )

    # "dp" is index 0 -> tensor_map value: 1 - 1 - 0 = 0
    expected_map = (0, -1, -1)

    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


def test_torch_embedding_layout_row_parallel():
    """
    Feature: PyTorch Embedding parallel
    Description: Row Parallel (Weight sharded on Vocab dim)
    Logic:
        Input [Batch, Seq] -> ("dp", "None")
        Weight [Vocab, Embed] -> ("mp", "None")

        According to the implementation logic provided:
        Output Indices = x_map + (w_map[-1],)
        x_map = ("dp", "None") -> (1, -1)
        w_map last dim = "None" -> -1
        Result -> (1, -1, -1) -> ("dp", "None", "None")

        Note: Real Row Parallel usually implies AllReduce or Partial output,
        but the provided implementation logic currently calculates pure sharding derivation.
    Expectation: Success based on current implementation logic
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    w_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout = w_layout("mp", "None") # Row Parallel

    output_layout = torch_op.infer_layout(
        [x_layout, w_layout],
        None
    )

    expected_map = (1, -1, -1) # Output embedding dim is not sharded

    assert output_layout.to_dict()["tensor_map"] == expected_map, (
        f"Test failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    )


def test_torch_embedding_inputs_check():
    """
    Feature: PyTorch Embedding parallel
    Description: Exception handling for insufficient inputs
    Expectation: ValueError
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

    with pytest.raises(ValueError, match="Embedding requires at least 2 layouts"):
        torch_op.infer_layout([x_layout], None)
