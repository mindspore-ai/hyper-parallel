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
"""parallel_scatter_update test"""

import pytest

from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_scatter_update import ScatterUpdateDistributedOp


def run_scenario(scenario_name, input_layout, indices_layout, updates_layout, expected_map):
    """Infer layout of ScatterUpdate operator"""
    print(f"\n{'=' * 80}")
    print(f"Test ScatterUpdate, Scenario: {scenario_name}")
    print('=' * 80)

    op = ScatterUpdateDistributedOp("ScatterUpdate")
    output_layout = op.infer_layout((input_layout, indices_layout, updates_layout), None)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"ScatterUpdate failed in scenario '{scenario_name}'. " \
        f"Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_scatter_update_data_parallel_1():
    """
    Feature: Data parallel on dimension 1.
    Description: input[1] sharded on dp, updates follows.
    Expectation: output matches input layout.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "None")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "dp", "None")

    run_scenario(
        "1. Data Parallel (DP) on dim 1",
        input_layout,
        indices_layout,
        updates_layout,
        expected_map=(-1, 2, -1)
    )


def test_scatter_update_model_parallel_2():
    """
    Feature: Model parallel on dimension 2.
    Description: input[2] sharded on mp, updates follows.
    Expectation: output matches input layout.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "None", "mp")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "None", "mp")

    run_scenario(
        "2. Model Parallel (MP) on dim 2",
        input_layout,
        indices_layout,
        updates_layout,
        expected_map=(-1, -1, 0)
    )


def test_scatter_update_hybrid_parallel_3():
    """
    Feature: Hybrid parallel on dimensions 1 and 2.
    Description: input[1] on dp, input[2] on mp, updates follows.
    Expectation: output matches input layout.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "mp")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "dp", "mp")

    run_scenario(
        "3. Hybrid Parallel (DP+MP)",
        input_layout,
        indices_layout,
        updates_layout,
        expected_map=(-1, 2, 0)
    )


def test_scatter_update_multi_dim_indices_4():
    """
    Feature: Multi-dimensional indices.
    Description: indices is 2D, updates first 2 dims cannot be sharded.
    Expectation: output matches input layout.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "mp")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None", "None")

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "None", "dp", "mp")

    run_scenario(
        "4. Multi-dimensional indices (2D)",
        input_layout,
        indices_layout,
        updates_layout,
        expected_map=(-1, 2, 0)
    )


def test_scatter_update_three_dim_input_5():
    """
    Feature: Three-dimensional input with complex sharding.
    Description: input[1] on cp, input[2] on mp, 3D indices.
    Expectation: output matches input layout.
    """
    mesh_shape = (2, 2, 2, 2)
    alias_name = ("dp", "cp", "mp", "tp")
    rank_list = list(range(16))

    input_layout = Layout(mesh_shape, alias_name, rank_list)
    input_layout = input_layout("None", "cp", "mp", "tp")

    indices_layout = Layout(mesh_shape, alias_name, rank_list)
    indices_layout = indices_layout("None", "None", "None")

    updates_layout = Layout(mesh_shape, alias_name, rank_list)
    updates_layout = updates_layout("None", "None", "None", "cp", "mp", "tp")

    run_scenario(
        "5. Three-dimensional input with 3D indices",
        input_layout,
        indices_layout,
        updates_layout,
        expected_map=(-1, 2, 1, 0)
    )


def test_scatter_update_replicate_all_6():
    """
    Feature: Full replication.
    Description: all tensors replicated across devices.
    Expectation: output is replicated.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "None", "None")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "None", "None")

    run_scenario(
        "6. Full replication",
        input_layout,
        indices_layout,
        updates_layout,
        expected_map=(-1, -1, -1)
    )


def test_scatter_update_error_input_first_dim_sharded_7():
    """
    Feature: Error case - input first dimension sharded.
    Description: input[0] cannot be sharded.
    Expectation: raise ValueError.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("dp", "None", "None")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "None", "None")

    with pytest.raises(ValueError, match="first dimension of input cannot be sharded"):
        run_scenario(
            "7. Error: input[0] sharded",
            input_layout,
            indices_layout,
            updates_layout,
            expected_map=None
        )


def test_scatter_update_error_indices_sharded_8():
    """
    Feature: Error case - indices sharded.
    Description: indices cannot be sharded.
    Expectation: raise ValueError.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "None")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("cp",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "dp", "None")

    with pytest.raises(ValueError, match="indices cannot be sharded"):
        run_scenario(
            "8. Error: indices sharded",
            input_layout,
            indices_layout,
            updates_layout,
            expected_map=None
        )


def test_scatter_update_error_updates_prefix_sharded_9():
    """
    Feature: Error case - updates prefix sharded.
    Description: updates first n dims (n=len(indices)) cannot be sharded.
    Expectation: raise ValueError.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "None")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("cp", "dp", "None")

    with pytest.raises(ValueError, match="dimensions of updates cannot be sharded"):
        run_scenario(
            "9. Error: updates prefix sharded",
            input_layout,
            indices_layout,
            updates_layout,
            expected_map=None
        )


def test_scatter_update_error_updates_mismatch_10():
    """
    Feature: Error case - updates sharding mismatch.
    Description: updates[indices_ndim:] must match input[1:].
    Expectation: raise ValueError.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "mp")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "mp", "dp")

    with pytest.raises(ValueError, match="updates sharding must match input"):
        run_scenario(
            "10. Error: updates sharding mismatch",
            input_layout,
            indices_layout,
            updates_layout,
            expected_map=None
        )


def test_scatter_update_error_updates_rank_mismatch_11():
    """
    Feature: Error case - updates rank mismatch.
    Description: updates rank must equal indices_ndim + input_ndim - 1.
    Expectation: raise ValueError.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "mp")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "dp")

    with pytest.raises(ValueError, match="updates rank mismatch"):
        run_scenario(
            "11. Error: updates rank mismatch",
            input_layout,
            indices_layout,
            updates_layout,
            expected_map=None
        )


def test_scatter_update_partial_propagation_12():
    """
    Feature: Partial status propagation.
    Description: input with partial status, output should inherit.
    Expectation: output layout has same partial status.
    """
    input_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    input_layout = input_layout("None", "dp", "mp")
    input_layout.set_partial_by_dev_axis("cp", "sum")

    indices_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    indices_layout = indices_layout("None",)

    updates_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    updates_layout = updates_layout("None", "dp", "mp")

    op = ScatterUpdateDistributedOp("ScatterUpdate")
    output_layout = op.infer_layout((input_layout, indices_layout, updates_layout), None)

    assert output_layout.get_partial_by_dev_id("cp") == "sum", \
        "Partial status should be propagated from input to output"
