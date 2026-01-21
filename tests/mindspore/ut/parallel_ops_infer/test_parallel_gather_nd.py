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
"""parallel_gather_nd test"""
import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_gather import GatherNdDistributedOp


def run_scenario(scenario_name, params_layout, indices_layout, expected_map=None,
                 extra_args=(), expect_error=False):
    """Infer layout of GatherNd operator"""
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    op = GatherNdDistributedOp("GatherNd")

    if expect_error:
        with pytest.raises(ValueError):
            op.infer_layout((params_layout, indices_layout), extra_args)
        return

    out_layout = op.infer_layout((params_layout, indices_layout), extra_args)
    assert out_layout.to_dict()["tensor_map"] == expected_map, (
        f"GatherNd failed. Expected {expected_map}, got {out_layout.to_dict()['tensor_map']}"
    )


def test_gathernd_data_parallel():
    """
    Feature: GatherNd layout inference.
    Description: Params is replicated; indices is sharded on the first dimension and replicated on the last dimension.
    Expectation: Output inherits sharding from indices[:-1].
    """
    mesh_shape = (8,)
    alias_name = ("dp",)
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)

    # params: [16, 64] -> cannot shard
    p_layout = layout("None", "None")

    # indices: [16, 2] -> shard dim0, last dim cannot shard
    i_layout = layout("dp", "None")

    run_scenario(
        "GatherNd data parallel",
        p_layout,
        i_layout,
        expected_map=(0,),
        extra_args=(([16, 64], [16, 2]),),
    )


def test_gathernd_model_parallel():
    """
    Feature: GatherNd layout inference.
    Description: Params is replicated; indices is sharded on the first dimension by model-parallel axis and replicated
                 on the last dimension.
    Expectation: Output inherits sharding from indices[:-1].
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)

    p_layout = layout("None", "None")
    i_layout = layout("mp", "None")

    # Under alias ("dp","mp"), "mp" -> tensor_map 0
    run_scenario(
        "GatherNd model parallel",
        p_layout,
        i_layout,
        expected_map=(0,),
        extra_args=(([16, 64], [16, 2]),),
    )


def test_gathernd_replicated():
    """
    Feature: GatherNd layout inference.
    Description: Params and indices are both replicated and indices last dimension is replicated.
    Expectation: Output is replicated.
    """
    mesh_shape = (8,)
    alias_name = ("dp",)
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)

    p_layout = layout("None", "None")
    i_layout = layout("None", "None")

    run_scenario(
        "GatherNd replicated",
        p_layout,
        i_layout,
        expected_map=(-1,),
        extra_args=(([16, 64], [16, 2]),),
    )


def test_gathernd_indices_last_dim_sharded_error():
    """
    Feature: GatherNd layout inference.
    Description: Indices last dimension is sharded, which is not allowed.
    Expectation: Raise ValueError.
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)

    p_layout = layout("None", "None")
    i_layout = layout("dp", "mp")  # last dim sharded -> illegal

    run_scenario(
        "GatherNd indices last dim sharded error",
        p_layout,
        i_layout,
        expect_error=True,
        extra_args=(([16, 64], [16, 2]),),
    )


def test_gathernd_params_sharded_error():
    """
    Feature: GatherNd layout inference.
    Description: Params is sharded, which is not allowed.
    Expectation: Raise ValueError.
    """
    mesh_shape = (8,)
    alias_name = ("dp",)
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)

    p_layout = layout("dp", "None")  # params sharded -> illegal
    i_layout = layout("dp", "None")  # indices ok

    run_scenario(
        "GatherNd params sharded error",
        p_layout,
        i_layout,
        expect_error=True,
        extra_args=(([16, 64], [16, 2]),),
    )


def test_gathernd_3d_indices_no_extra_args():
    """
    Feature: GatherNd layout inference.
    Description: 3D indices without extra_args.
    Expectation: Output inherits only indices[:-1] sharding.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)

    p_layout = layout("None", "None", "None")
    i_layout = layout("dp", "cp", "None")

    run_scenario(
        "GatherNd 3D indices no extra args (assume full indexing)",
        p_layout,
        i_layout,
        expected_map=(2, 1, -1),
        extra_args=(([16, 64, 32], [16, 8, 2]),),
    )


def test_gathernd_params_is_none_layout():
    """
    Feature: GatherNd layout inference.
    Description: Params layout is None (plain Tensor), indices is sharded.
    Expectation: Output inherits sharding from indices[:-1].
    """
    mesh_shape = (8,)
    alias_name = ("dp",)
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)

    # params has no layout (None) - simulating plain Tensor
    p_layout = None
    i_layout = layout("dp", "None")

    run_scenario(
        "GatherNd params layout is None",
        p_layout,
        i_layout,
        expected_map=(0,),
        extra_args=(([16, 64], [16, 2]),),
    )
