# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""test sub_mesh"""
from tests.common.parallel_case import parallel_run, TorchCase
from tests.common.mark_utils import arg_mark

SUBMESH = "submesh.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_submesh_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_full_mesh_shard_forward_1
        2.test_sub_mesh_column_parallel_forward
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_full_mesh_shard_forward_1", 18311, 4),
        TorchCase(SUBMESH, "test_sub_mesh_column_parallel_forward", 18312, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_full_mesh_shard_forward_2
        2.test_sub_mesh_row_parallel_forward
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_full_mesh_shard_forward_2", 11754, 4),
        TorchCase(SUBMESH, "test_sub_mesh_row_parallel_forward", 11755, 4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_submesh_group2_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_full_mesh_shard_forward_2
        2.test_sub_mesh_row_parallel_forward
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_full_mesh_shard_forward_2", num_proc=4),
        TorchCase(SUBMESH, "test_sub_mesh_row_parallel_forward", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_row_parallel_redistribute_forward
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_row_parallel_redistribute_forward", 11756, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group4():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_1
        2.test_sub_mesh_redistribute_2
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_1", 11757, 4),
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_2", 11758, 4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_submesh_group4_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_1
        2.test_sub_mesh_redistribute_2
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_1", num_proc=4),
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_2", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group5():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_3
        2.test_sub_mesh_redistribute_4
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_3", 11759, 2),
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_4", 11760, 2),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_submesh_group5_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_3
        2.test_sub_mesh_redistribute_4
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_3", num_proc=2),
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_4", num_proc=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_5():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_5
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_5", 11761, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_6():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_6
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_6", 11762, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_7():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_7
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_7", 11763, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_8():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_8
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_8", 11764, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_9():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_9
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_9", 11765, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_10():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_10
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_10", 11766, 8)
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_10_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sub_mesh_redistribute_10
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(SUBMESH, "test_sub_mesh_redistribute_10", num_proc=8)
    ])
