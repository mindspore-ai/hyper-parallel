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
"""test sub_mesh"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

SUBMESH = "submesh.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group1():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_full_mesh_shard_forward_1
        2. test_sub_mesh_column_parallel_forward
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_full_mesh_shard_forward_1", 11654, 4, 4),
        MindSporeCase(SUBMESH, "test_sub_mesh_column_parallel_forward", 11655, 4, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_submesh_group2():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_full_mesh_shard_forward_2
        2. test_sub_mesh_row_parallel_forward
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_full_mesh_shard_forward_2", 18313, 4, 4),
        MindSporeCase(SUBMESH, "test_sub_mesh_row_parallel_forward", 11656, 4, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group3():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_row_parallel_redistribute_forward
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_row_parallel_redistribute_forward", 18314, 4, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_submesh_group4():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_1", 18315, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group5():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_2", 18316, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group6():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_3", 18317, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group7():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_4", 18318, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group8():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_5", 11657, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group9():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_6
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_6", 18319, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group10():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_7
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_7", 18320, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group11():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_8
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_8", 18321, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group12():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_9
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_9", 18322, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_submesh_group13():
    """
    Feature: parallel run case in submesh
    Description:
        1. test_sub_mesh_redistribute_10
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SUBMESH, "test_sub_mesh_redistribute_10", 18323, 8, 8),
    ])
