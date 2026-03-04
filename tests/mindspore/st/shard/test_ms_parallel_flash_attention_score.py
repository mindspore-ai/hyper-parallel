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
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_bsh_replicate():
    """
    Feature: test flash_attention_score with BSH layout, no parallelism.
    Description: All tensors replicated, verify basic DTensor pass-through.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_replicate"
    master_port = 11010
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_dp():
    """
    Feature: test flash_attention_score with BSH layout, data parallelism.
    Description: Shard on batch dimension, gather and compare with standalone.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_dp"
    master_port = 11011
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_mp():
    """
    Feature: test flash_attention_score with BSH layout, head parallelism.
    Description: Shard on hidden dimension, head_num adjusted correctly.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_mp"
    master_port = 11012
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_sp():
    """
    Feature: test flash_attention_score with BSH layout, sequence parallelism.
    Description: Q shard on seq dim, KV replicated, sparse params adjusted.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_sp"
    master_port = 11013
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_dp_mp_2d():
    """
    Feature: test flash_attention_score with BSH layout, DP + MP on 2D mesh.
    Description: Shard batch and hidden on (4, 2) mesh.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_dp_mp_2d"
    master_port = 11014
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_sp_mp_2d():
    """
    Feature: test flash_attention_score with BSH layout, SP + MP on 2D mesh.
    Description: Q shard on seq + hidden, KV replicate seq + shard hidden.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_sp_mp_2d"
    master_port = 11015
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_dp_sp_mp_3d():
    """
    Feature: test flash_attention_score with BSH layout, DP + SP + MP on 3D mesh.
    Description: Full parallelism on (2, 2, 2) mesh.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_dp_sp_mp_3d"
    master_port = 11016
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_bnsd_dp_mp():
    """
    Feature: test flash_attention_score with BNSD layout, DP + MP.
    Description: Shard batch and head on (4, 2) mesh.
    Expectation: Correct local output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bnsd_dp_mp"
    master_port = 11017
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_bnsd_sp():
    """
    Feature: test flash_attention_score with BNSD layout, DP + SP.
    Description: Q shard on seq dim 2, KV replicate on seq.
    Expectation: Correct local output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bnsd_sp"
    master_port = 11018
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sbh_dp():
    """
    Feature: test flash_attention_score with SBH layout, data parallelism.
    Description: Shard on batch dim (dim 1 in SBH).
    Expectation: Correct local output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_sbh_dp"
    master_port = 11019
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsnd_dp_mp():
    """
    Feature: test flash_attention_score with BSND layout, DP + MP.
    Description: Shard batch and head on (4, 2) mesh.
    Expectation: Correct local output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsnd_dp_mp"
    master_port = 11020
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp():
    """
    Feature: test flash_attention_score with TND layout, data parallelism.
    Description: Shard T-dim with cumulative actual_seq_qlen/kvlen.
    Expectation: Correct local output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_tnd_dp"
    master_port = 11021
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_tnd_mp():
    """
    Feature: test flash_attention_score with TND layout, head parallelism.
    Description: Shard N-dim, head_num adjusted.
    Expectation: Correct local output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_tnd_mp"
    master_port = 11022
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp_mp():
    """
    Feature: test flash_attention_score with TND layout, DP + MP on 2D mesh.
    Description: Shard T-dim and N-dim on (4, 2) mesh.
    Expectation: Correct local output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_tnd_dp_mp"
    master_port = 11023
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_0():
    """
    Feature: test flash_attention_score SP with sparse_mode=0.
    Description: Sequence parallel with default mask, no pre/next token adjustment.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_sp_sparse_mode_0"
    master_port = 11024
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_2():
    """
    Feature: test flash_attention_score SP with sparse_mode=2.
    Description: Sequence parallel with left_up_causal mask, offset via LEFT_UP_TO_RIGHT_DOWN.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_sp_sparse_mode_2"
    master_port = 11025
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_3():
    """
    Feature: test flash_attention_score SP with sparse_mode=3.
    Description: Sequence parallel with right_down_causal mask, offset via RIGHT_DOWN_TO_RIGHT_DOWN.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_sp_sparse_mode_3"
    master_port = 11026
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_4():
    """
    Feature: test flash_attention_score SP with sparse_mode=4.
    Description: Sequence parallel with band mask and custom pre_tokens/next_tokens.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_sp_sparse_mode_4"
    master_port = 11027
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dp_sparse_mode_1():
    """
    Feature: test flash_attention_score DP with sparse_mode=1.
    Description: Data parallel with all_mask, requires attn_mask provided.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_dp_sparse_mode_1"
    master_port = 11028
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level4", card_mark="allcards", essential_mark="essential")
def test_dp_sparse_mode_4():
    """
    Feature: test flash_attention_score DP with sparse_mode=4.
    Description: Data parallel with band mask and custom pre_tokens/next_tokens.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_dp_sparse_mode_4"
    master_port = 11029
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_bsh_custom_scale():
    """
    Feature: test flash_attention_score with custom scalar_value.
    Description: BSH DP with scalar_value=0.125 instead of default.
    Expectation: Output matches standalone execution with same scalar_value.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_custom_scale"
    master_port = 11030
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_dropout():
    """
    Feature: test flash_attention_score with dropout.
    Description: BSH DP with keep_prob=0.9, smoke test due to randomness.
    Expectation: Correct output shape without crash.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_dropout"
    master_port = 11031
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_long_sequence_sp():
    """
    Feature: test flash_attention_score SP with long sequence.
    Description: BSH SP with seq_len=2048.
    Expectation: Correct output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_long_sequence_sp"
    master_port = 11032
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_large_batch_dp():
    """
    Feature: test flash_attention_score DP with large batch.
    Description: BSH DP with batch_size=64.
    Expectation: Correct output shape.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_large_batch_dp"
    master_port = 11033
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_redistribute_then_attention():
    """
    Feature: test flash_attention_score after redistribute.
    Description: Redistribute from Replicate to Shard before running attention.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bsh_redistribute_then_attention"
    master_port = 11034
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_sp_sparse_mode_2_with_2way_split():
    """
    Feature: test flash_attention_score SP sparse_mode=2 with 2-way split.
    Description: Verify LEFT_UP_TO_RIGHT_DOWN offset on (2, 4) mesh.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_sp_sparse_mode_2_with_2way_split"
    master_port = 11035
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sp_sparse_mode_3_with_2way_split():
    """
    Feature: test flash_attention_score SP sparse_mode=3 with 2-way split.
    Description: Verify RIGHT_DOWN_TO_RIGHT_DOWN offset on (2, 4) mesh.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_sp_sparse_mode_3_with_2way_split"
    master_port = 11036
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_bnsd_sp_correctness():
    """
    Feature: test flash_attention_score BNSD SP correctness.
    Description: BNSD layout with Q seq shard, KV replicate, compare with standalone.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_bnsd_sp_correctness"
    master_port = 11037
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp_correctness():
    """
    Feature: test flash_attention_score TND DP correctness with actual_seq_len adjustment.
    Description: TND layout with 8-way DP on T dimension, verifies pure DP correctly
        adjusts actual_seq_qlen/actual_seq_kvlen via clamp while skipping sparse params
        adjustment. Gather output and compare with standalone.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_tnd_dp_correctness"
    master_port = 11038
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_tnd_cp():
    """
    Feature: test flash_attention_score TND layout with context parallelism.
    Description: Q Shard(0) on cp, KV Replicate, sparse_mode=3 required by CP.
        actual_seq_qlen/kvlen adjusted via clamp logic for s1_split_num > 1.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_tnd_cp"
    master_port = 11039
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp_kv_sharded():
    """
    Feature: test flash_attention_score TND DP with kv_is_sharded=True.
    Description: TND layout with Q/K/V all Shard(0) on T dimension, triggers
        kv_is_sharded clamp branch in actual_seq_len adjustment.
    Expectation: Output matches standalone execution.
    """
    glog_v = 2
    file_name = "ms_flash_attention_score_shard_in_python.py"
    case_name = "test_tnd_dp_kv_sharded"
    master_port = 11040
    msrun_case(glog_v, file_name, case_name, master_port)
