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
"""test parallel flash attention score"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_FLASH_ATTENTION_SCORE = "parallel_op_flash_attention_score.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_parallel_flash_attention_score_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bsh_replicate
        2.test_bsh_dp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_replicate", 11010, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_dp", 11011, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bsh_dp_mp_2d
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_dp_mp_2d", 11014, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bnsd_dp_mp
        2.test_bnsd_sp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bnsd_dp_mp", 11017, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bnsd_sp", 11018, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group4():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sp_sparse_mode_2
        2.test_error_kv_strategy_mismatch
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_sp_sparse_mode_2", 11025, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_kv_strategy_mismatch", 11030, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group5():
    """
    Feature: parallel run case in shard
    Description:
        1.test_error_head_num_not_divisible
        2.test_error_kv_seq_sharding_blocked
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_head_num_not_divisible", 11031, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_kv_seq_sharding_blocked", 11032, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group6():
    """
    Feature: parallel run case in shard
    Description:
        1.test_error_sparse_mode_1_no_mask
        2.test_error_tnd_sp_without_actual_seq_len
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_sparse_mode_1_no_mask", 11033, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_tnd_sp_without_actual_seq_len", 11034, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group7():
    """
    Feature: parallel run case in shard
    Description:
        1.test_error_bnsd_kv_seq_sharding_blocked
        2.test_bsh_custom_scale
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_bnsd_kv_seq_sharding_blocked", 11035, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_custom_scale", 11036, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group8():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bsh_long_sequence_sp
        2.test_sp_sparse_mode_2_with_2way_split
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_long_sequence_sp", 11038, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_sp_sparse_mode_2_with_2way_split", 11041, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group9():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bsh_large_batch_dp
        2.test_bnsd_sp_correctness
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_large_batch_dp", 11039, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bnsd_sp_correctness", 11043, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group10():
    """
    Feature: parallel run case in shard
    Description:
        1.test_error_batch_sharding_mismatch
        2.test_error_hidden_sharding_mismatch
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_batch_sharding_mismatch", 11048, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_hidden_sharding_mismatch", 11049, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group11():
    """
    Feature: parallel run case in shard
    Description:
        1.test_error_dim_sharding_mismatch
        2.test_error_tnd_cp_wrong_sparse_mode
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_dim_sharding_mismatch", 11050, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_tnd_cp_wrong_sparse_mode", 11051, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group12():
    """
    Feature: parallel run case in shard
    Description:
        1.test_error_tnd_cp_qk_global_t_mismatch
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_error_tnd_cp_qk_global_t_mismatch", 11052, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group13():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bsh_mp
        2.test_bsh_sp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_mp", 11012, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_sp", 11013, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group14():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bsh_sp_mp_2d
        2.test_bsh_dp_sp_mp_3d
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_sp_mp_2d", 11015, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_dp_sp_mp_3d", 11016, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group15():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sbh_dp
        2.test_bsnd_dp_mp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_sbh_dp", 11019, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsnd_dp_mp", 11020, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group16():
    """
    Feature: parallel run case in shard
    Description:
        1.test_tnd_dp
        2.test_tnd_mp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_tnd_dp", 11021, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_tnd_mp", 11022, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group17():
    """
    Feature: parallel run case in shard
    Description:
        1.test_tnd_dp_mp
        2.test_sp_sparse_mode_0
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_tnd_dp_mp", 11023, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_sp_sparse_mode_0", 11024, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group18():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sp_sparse_mode_3
        2.test_sp_sparse_mode_4
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_sp_sparse_mode_3", 11026, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_sp_sparse_mode_4", 11027, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group19():
    """
    Feature: parallel run case in shard
    Description:
        1.test_dp_sparse_mode_1
        2.test_dp_sparse_mode_4
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_dp_sparse_mode_1", 11028, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_dp_sparse_mode_4", 11029, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group20():
    """
    Feature: parallel run case in shard
    Description:
        1.test_bsh_dropout
        2.test_bsh_redistribute_then_attention
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_dropout", 11037, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_bsh_redistribute_then_attention", 11040, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_flash_attention_score_group21():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sp_sparse_mode_3_with_2way_split
        2.test_tnd_dp_correctness
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_sp_sparse_mode_3_with_2way_split", 11042, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_tnd_dp_correctness", 11044, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group22():
    """
    Feature: parallel run case in shard
    Description:
        1.test_tnd_cp
        2.test_fa_vs_sdpa_cross_validation
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_tnd_cp", 11045, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_fa_vs_sdpa_cross_validation", 11046, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_flash_attention_score_group23():
    """
    Feature: parallel run case in shard
    Description:
        1.test_fa_vs_sdpa_distributed_cross_validation
        2.test_tnd_dp_kv_sharded
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_fa_vs_sdpa_distributed_cross_validation", 11047, 4),
        TorchCase(PARALLEL_FLASH_ATTENTION_SCORE, "test_tnd_dp_kv_sharded", 11053, 4),
    ])
