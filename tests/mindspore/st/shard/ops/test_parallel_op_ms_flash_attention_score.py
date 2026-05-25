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

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_ms_flash_attention_score.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_ms_flash_attention_score_group1():
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
        MindSporeCase(IMPL_FILE, "test_bsh_replicate", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsh_dp", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsh_sp", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_sp_sparse_mode_0", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_ms_flash_attention_score_group2():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_mp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_bsh_mp", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_ms_flash_attention_score_group3():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_dp_mp_2d
        2. test_bsh_sp_mp_2d
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_bsh_dp_mp_2d", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsh_sp_mp_2d", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_ms_flash_attention_score_group4():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bnsd_dp_mp
        2. test_bnsd_sp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_bnsd_dp_mp", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bnsd_sp", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_ms_flash_attention_score_group5():
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
        MindSporeCase(IMPL_FILE, "test_sp_sparse_mode_2", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_sp_sparse_mode_3", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_sp_sparse_mode_4", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_dp_sparse_mode_1", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_ms_flash_attention_score_group6():
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
        MindSporeCase(IMPL_FILE, "test_dp_sparse_mode_4", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsh_custom_scale", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsh_redistribute_then_attention", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bnsd_sp_correctness", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_ms_flash_attention_score_group7():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_sp_sparse_mode_2_with_2way_split
        2. test_tnd_dp_correctness
        3. test_tnd_cp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_sp_sparse_mode_2_with_2way_split", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_tnd_dp_correctness", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_tnd_cp", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_ms_flash_attention_score_group8():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_tnd_dp_kv_sharded
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_tnd_dp_kv_sharded", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_ms_flash_attention_score_group9():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_dp_sp_mp_3d
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_bsh_dp_sp_mp_3d", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_ms_flash_attention_score_group10():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_sbh_dp
        2. test_bsnd_dp_mp
        3. test_tnd_dp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_sbh_dp", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsnd_dp_mp", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_tnd_dp", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_ms_flash_attention_score_group11():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_tnd_mp
        2. test_tnd_dp_mp
        3. test_bsh_dropout
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_tnd_mp", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_tnd_dp_mp", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsh_dropout", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_ms_flash_attention_score_group12():
    """
    Feature: parallel run case in ms_flash_attention_score_shard_in_python
    Description:
        1. test_bsh_long_sequence_sp
        2. test_bsh_large_batch_dp
        3. test_sp_sparse_mode_3_with_2way_split
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_bsh_long_sequence_sp", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_bsh_large_batch_dp", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_sp_sparse_mode_3_with_2way_split", worker_num=4, local_worker_num=4, glog_v=2),
    ])
