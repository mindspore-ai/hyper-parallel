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
"""test gatherd shard"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

GATHER_D_SHARD_IN_PYTHON = "gather_d_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gatherd_data_parallel_dim0_1():
    '''
    Feature: GatherD operator.
    Description: Test GatherD dim0 batch-sharded input with replicated index in python shard.
    Expectation: Run success.
    '''
    parallel_run([
        MindSporeCase(GATHER_D_SHARD_IN_PYTHON, "test_gatherd_data_parallel_dim0_1", 11400, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gatherd_data_parallel_dim1_2():
    '''
    Feature: GatherD operator.
    Description: Test GatherD dim1 sequence-sharded input with replicated index in python shard.
    Expectation: Run success.
    '''
    parallel_run([
        MindSporeCase(GATHER_D_SHARD_IN_PYTHON, "test_gatherd_data_parallel_dim1_2", 11401, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gatherd_row_parallel_3():
    '''
    Feature: GatherD operator.
    Description: Test GatherD row-parallel runtime case in python shard.
    Expectation: Run success.
    '''
    parallel_run([
        MindSporeCase(GATHER_D_SHARD_IN_PYTHON, "test_gatherd_row_parallel_3", 11402, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gatherd_input_multi_shard_4():
    '''
    Feature: GatherD operator.
    Description: Test GatherD matched non-dim shard runtime case in python shard.
    Expectation: Run success.
    '''
    parallel_run([
        MindSporeCase(GATHER_D_SHARD_IN_PYTHON, "test_gatherd_input_multi_shard_4", 11403, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gatherd_3d_mesh_input_multi_shard_5():
    '''
    Feature: GatherD operator.
    Description: Test GatherD 3D-mesh gather-axis sharding runtime case in python shard.
    Expectation: Run success.
    '''
    parallel_run([
        MindSporeCase(GATHER_D_SHARD_IN_PYTHON, "test_gatherd_3d_mesh_input_multi_shard_5", 11404, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gatherd_3d_mesh_matched_non_dim_shard_6():
    '''
    Feature: GatherD operator.
    Description: Test GatherD 3D-mesh matched non-dim shard runtime case in python shard.
    Expectation: Run success.
    '''
    parallel_run([
        MindSporeCase(GATHER_D_SHARD_IN_PYTHON, "test_gatherd_3d_mesh_matched_non_dim_shard_6", 11405, 8, 8, 2),
    ])
