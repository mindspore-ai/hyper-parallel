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
