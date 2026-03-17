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

"""Test flash_attention_score distributed operator"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON = "ms_flash_attention_score_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_flash_attention_score_shard_in_python_group1():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_replicate
        2. test_bsh_dp
        3. test_bsh_sp
        4. test_sp_sparse_mode_0
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_replicate", 11010, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_dp", 11011, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_sp", 11013, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_sp_sparse_mode_0", 11024, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_flash_attention_score_shard_in_python_group2():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_mp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_mp", 11012, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_flash_attention_score_shard_in_python_group3():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_dp_mp_2d
        2. test_bsh_sp_mp_2d
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_dp_mp_2d", 11014, 4, 4, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_sp_mp_2d", 11015, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_flash_attention_score_shard_in_python_group4():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bnsd_dp_mp
        2. test_bnsd_sp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bnsd_dp_mp", 11017, 4, 4, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bnsd_sp", 11018, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_ms_flash_attention_score_shard_in_python_group5():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_sp_sparse_mode_2
        2. test_sp_sparse_mode_3
        3. test_sp_sparse_mode_4
        4. test_dp_sparse_mode_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_sp_sparse_mode_2", 11025, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_sp_sparse_mode_3", 11026, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_sp_sparse_mode_4", 11027, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_dp_sparse_mode_1", 11028, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_flash_attention_score_shard_in_python_group6():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_dp_sparse_mode_4
        2. test_bsh_custom_scale
        3. test_bsh_redistribute_then_attention
        4. test_bnsd_sp_correctness
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_dp_sparse_mode_4", 11029, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_custom_scale", 11030, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_redistribute_then_attention", 11034, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bnsd_sp_correctness", 11037, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_ms_flash_attention_score_shard_in_python_group7():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_sp_sparse_mode_2_with_2way_split
        2. test_tnd_dp_correctness
        3. test_tnd_cp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_sp_sparse_mode_2_with_2way_split", 11035, 4, 4,
                      2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_tnd_dp_correctness", 11038, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_tnd_cp", 11039, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_ms_flash_attention_score_shard_in_python_group8():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_tnd_dp_kv_sharded
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_tnd_dp_kv_sharded", 11040, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_ms_flash_attention_score_shard_in_python_group9():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_dp_sp_mp_3d
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_dp_sp_mp_3d", 11016, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_ms_flash_attention_score_shard_in_python_group10():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_sbh_dp
        2. test_bsnd_dp_mp
        3. test_tnd_dp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_sbh_dp", 11019, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsnd_dp_mp", 11020, 4, 4, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_tnd_dp", 11021, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_ms_flash_attention_score_shard_in_python_group11():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_tnd_mp
        2. test_tnd_dp_mp
        3. test_bsh_dropout
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_tnd_mp", 11022, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_tnd_dp_mp", 11023, 4, 4, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_dropout", 11031, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_ms_flash_attention_score_shard_in_python_group12():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_long_sequence_sp
        2. test_bsh_large_batch_dp
        3. test_sp_sparse_mode_3_with_2way_split
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_long_sequence_sp", 11032, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_bsh_large_batch_dp", 11033, 2, 2, 2),
        MindSporeCase(MS_FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_sp_sparse_mode_3_with_2way_split", 11036, 4, 4,
                      2),
    ])
