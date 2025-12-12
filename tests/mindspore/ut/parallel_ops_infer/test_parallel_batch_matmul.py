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

from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_matmul import BatchMatMulDistributedOp


def run_scenario(scenario_name, x_layout, w_layout, expected_map, transpose_a=False, transpose_b=False):
    """Infer layout of BatchMatMul"""
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print('=' * 80)

    op = BatchMatMulDistributedOp("BatchMatMul")
    output_layout = op.infer_layout((x_layout, w_layout), (transpose_a, transpose_b))
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Test BatchMatMul failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


base_device_matrix = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_bmm_tensor_parallel():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    x_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout4 = x_layout4("dp", "None", "mp")

    w_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout4 = w_layout4("None", "mp", "None")

    run_scenario(
        "4. Tensor Model Parallel (TMP)",
        x_layout4,
        w_layout4,
        expected_map=(2, -1, -1)
    )


def test_bmm_transpose_tensor_parallel():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard, transpose=True.
    Expectation: Run success.
    '''
    x_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout4 = x_layout4("dp", "mp", "None")

    w_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout4 = w_layout4("None", "mp", "None")

    run_scenario(
        "4. Tensor Model Parallel (TMP)",
        x_layout4,
        w_layout4,
        expected_map=(2, -1, -1),
        transpose_a=True,
        transpose_b=False
    )
