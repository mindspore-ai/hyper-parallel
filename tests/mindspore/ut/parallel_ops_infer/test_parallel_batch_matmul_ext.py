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
"""parallel_batch_matmul_ext test"""

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_matmul import BatchMatMulExtDistributedOp


def run_scenario(scenario_name, x_layout, w_layout, expected_map):
    """Infer layout of BatchMatMul"""
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print('=' * 80)

    op = BatchMatMulExtDistributedOp("BatchMatMulExt")
    output_layout = op.infer_layout((x_layout, w_layout))
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Test BatchMatMulExt failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_bmm_ext_data_parallel():
    '''
    Feature: Data parallel in python shard.
    Description: Test data parallel in python shard.
    Expectation: Run success.
    '''
    x_layout1 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout1 = x_layout1("dp", "None", "None")

    w_layout1 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout1 = w_layout1("dp", "None", "None")

    run_scenario(
        "1. Data Parallel (DP)",
        x_layout1,
        w_layout1,
        expected_map=(2, -1, -1)
    )


def test_bmm_ext_model_parallel():
    '''
    Feature: Model parallel in python shard.
    Description: Test model parallel in python shard.
    Expectation: Run success.
    '''
    x_layout2 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout2 = x_layout2("None", "None", "None")

    w_layout2 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout2 = w_layout2("None", "None", "mp")

    run_scenario(
        "2. Model Parallel (MP)",
        x_layout2,
        w_layout2,
        expected_map=(-1, -1, 0)
    )


def test_bmm_ext_hybrid_parallel():
    '''
    Feature: Hybrid parallel in python shard.
    Description: Test hybrid parallel in python shard.
    Expectation: Run success.
    '''
    x_layout3 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout3 = x_layout3("dp", "cp", "None")

    w_layout3 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout3 = w_layout3("dp", "None", "mp")

    run_scenario(
        "3. Hybrid Parallel (DP + CP + MP)",
        x_layout3,
        w_layout3,
        expected_map=(2, 1, 0)
    )


def test_bmm_ext_tensor_parallel():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    x_layout4 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout4 = x_layout4("dp", "None", "mp")

    w_layout4 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout4 = w_layout4("None", "mp", "None")

    run_scenario(
        "4. Tensor Model Parallel (TMP)",
        x_layout4,
        w_layout4,
        expected_map=(2, -1, -1)
    )


def test_bmm_ext_hybrid_tensor_parallel():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    x_layout5 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout5 = x_layout5("dp", "cp", "mp")

    w_layout5 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout5 = w_layout5("None", "mp", "None")

    run_scenario(
        "5. Hybrid Tensor Model Parallel (TMP)",
        x_layout5,
        w_layout5,
        expected_map=(2, 1, -1)
    )


def test_bmm_ext_multi_shard_tensor_parallel():
    '''
    Feature: Multi shard tensor parallel in python shard.
    Description: Test multi shard tensor parallel in python shard.
    Expectation: Run success.
    '''
    x_layout6 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout6 = x_layout6("dp", "None", "mp")

    w_layout6 = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    w_layout6 = w_layout6("dp", "mp", "cp")

    run_scenario(
        "6. Multi shard Tensor Model Parallel (TMP)",
        x_layout6,
        w_layout6,
        expected_map=(2, -1, 1)
    )


def test_bmm_ext_multi_shard_one_dim_tensor_parallel():
    '''
    Feature: Multi shard one dim tensor parallel in python shard.
    Description: Test Multi shard one dim tensor parallel in python shard.
    Expectation: Run success.
    '''
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "tp", "mp")
    rank_list = list(range(8))
    x_layout7 = Layout(mesh_shape, alias_name, rank_list)
    x_layout7 = x_layout7("None", ("dp", "tp"), "None")

    w_layout7 = Layout(mesh_shape, alias_name, rank_list)
    w_layout7 = w_layout7("None", "None", "mp")

    run_scenario(
        "7. Multi shard in one dim Tensor Model Parallel (TMP)",
        x_layout7,
        w_layout7,
        expected_map=(-1, (2, 1), 0)
    )
