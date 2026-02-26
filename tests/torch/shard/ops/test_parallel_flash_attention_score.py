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
"""Test npu_fusion_attention distributed operator"""

from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_bsh_replicate():
    """
    Feature: test npu_fusion_attention with BSH layout, no parallelism.
    Description: All tensors replicated, verify basic DTensor pass-through.
    Expectation: Output matches standalone execution.
    """
    master_port = 11010
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_replicate"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_dp():
    """
    Feature: test npu_fusion_attention with BSH layout, data parallelism.
    Description: Shard on batch dimension, gather and compare with standalone.
    Expectation: Output matches standalone execution.
    """
    master_port = 11011
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_dp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_mp():
    """
    Feature: test npu_fusion_attention with BSH layout, head parallelism.
    Description: Shard on hidden dimension, head_num adjusted correctly.
    Expectation: Output matches standalone execution.
    """
    master_port = 11012
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_mp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_sp():
    """
    Feature: test npu_fusion_attention with BSH layout, sequence parallelism.
    Description: Q shard on seq dim, KV replicated, sparse params adjusted.
    Expectation: Output matches standalone execution.
    """
    master_port = 11013
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_sp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_dp_mp_2d():
    """
    Feature: test npu_fusion_attention with BSH layout, DP + MP on 2D mesh.
    Description: Shard batch and hidden on (4, 2) mesh.
    Expectation: Output matches standalone execution.
    """
    master_port = 11014
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_dp_mp_2d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_bsh_sp_mp_2d():
    """
    Feature: test npu_fusion_attention with BSH layout, SP + MP on 2D mesh.
    Description: Q shard on seq + hidden, KV replicate seq + shard hidden.
    Expectation: Output matches standalone execution.
    """
    master_port = 11015
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_sp_mp_2d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_dp_sp_mp_3d():
    """
    Feature: test npu_fusion_attention with BSH layout, DP + SP + MP on 3D mesh.
    Description: Full parallelism on (2, 2, 2) mesh.
    Expectation: Output matches standalone execution.
    """
    master_port = 11016
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_dp_sp_mp_3d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bnsd_dp_mp():
    """
    Feature: test npu_fusion_attention with BNSD layout, DP + MP.
    Description: Shard batch and head on (4, 2) mesh.
    Expectation: Correct local output shape.
    """
    master_port = 11017
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bnsd_dp_mp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bnsd_sp():
    """
    Feature: test npu_fusion_attention with BNSD layout, DP + SP.
    Description: Q shard on seq dim 2, KV replicate on seq.
    Expectation: Correct local output shape.
    """
    master_port = 11018
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bnsd_sp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sbh_dp():
    """
    Feature: test npu_fusion_attention with SBH layout, data parallelism.
    Description: Shard on batch dim (dim 1 in SBH).
    Expectation: Correct local output shape.
    """
    master_port = 11019
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_sbh_dp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsnd_dp_mp():
    """
    Feature: test npu_fusion_attention with BSND layout, DP + MP.
    Description: Shard batch and head on (4, 2) mesh.
    Expectation: Correct local output shape.
    """
    master_port = 11020
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsnd_dp_mp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp():
    """
    Feature: test npu_fusion_attention with TND layout, data parallelism.
    Description: Shard T-dim with cumulative actual_seq_qlen/kvlen.
    Expectation: Correct local output shape.
    """
    master_port = 11021
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_tnd_dp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_tnd_mp():
    """
    Feature: test npu_fusion_attention with TND layout, head parallelism.
    Description: Shard N-dim, head_num adjusted.
    Expectation: Correct local output shape.
    """
    master_port = 11022
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_tnd_mp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp_mp():
    """
    Feature: test npu_fusion_attention with TND layout, DP + MP on 2D mesh.
    Description: Shard T-dim and N-dim on (4, 2) mesh.
    Expectation: Correct local output shape.
    """
    master_port = 11023
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_tnd_dp_mp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_0():
    """
    Feature: test npu_fusion_attention SP with sparse_mode=0.
    Description: Sequence parallel with default mask, no pre/next token adjustment.
    Expectation: Output matches standalone execution.
    """
    master_port = 11024
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_sp_sparse_mode_0"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_2():
    """
    Feature: test npu_fusion_attention SP with sparse_mode=2.
    Description: Sequence parallel with left_up_causal mask, offset via LEFT_UP_TO_RIGHT_DOWN.
    Expectation: Output matches standalone execution.
    """
    master_port = 11025
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_sp_sparse_mode_2"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_3():
    """
    Feature: test npu_fusion_attention SP with sparse_mode=3.
    Description: Sequence parallel with right_down_causal mask, offset via RIGHT_DOWN_TO_RIGHT_DOWN.
    Expectation: Output matches standalone execution.
    """
    master_port = 11026
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_sp_sparse_mode_3"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sp_sparse_mode_4():
    """
    Feature: test npu_fusion_attention SP with sparse_mode=4.
    Description: Sequence parallel with band mask and custom pre/next_tockens.
    Expectation: Output matches standalone execution.
    """
    master_port = 11027
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_sp_sparse_mode_4"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dp_sparse_mode_1():
    """
    Feature: test npu_fusion_attention DP with sparse_mode=1.
    Description: Data parallel with all_mask, requires atten_mask provided.
    Expectation: Output matches standalone execution.
    """
    master_port = 11028
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_dp_sparse_mode_1"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dp_sparse_mode_4():
    """
    Feature: test npu_fusion_attention DP with sparse_mode=4.
    Description: Data parallel with band mask and custom pre/next_tockens.
    Expectation: Output matches standalone execution.
    """
    master_port = 11029
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_dp_sparse_mode_4"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_kv_strategy_mismatch():
    """
    Feature: test npu_fusion_attention rejects mismatched KV strategies.
    Description: Key and Value have different sharding strategies.
    Expectation: Raise ValueError.
    """
    master_port = 11030
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_kv_strategy_mismatch"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_head_num_not_divisible():
    """
    Feature: test npu_fusion_attention rejects non-divisible head_num.
    Description: head_num=17 is not divisible by head_split_num=8.
    Expectation: Raise ValueError.
    """
    master_port = 11031
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_head_num_not_divisible"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_kv_seq_sharding_blocked():
    """
    Feature: test npu_fusion_attention blocks KV sequence sharding for BSH.
    Description: Both Q and KV sharded on seq dim without Ring Attention.
    Expectation: Raise NotImplementedError.
    """
    master_port = 11032
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_kv_seq_sharding_blocked"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_sparse_mode_1_no_mask():
    """
    Feature: test npu_fusion_attention rejects sparse_mode=1 without mask.
    Description: sparse_mode=1 (allMask) requires atten_mask to be provided.
    Expectation: Raise ValueError.
    """
    master_port = 11033
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_sparse_mode_1_no_mask"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_tnd_sp_without_actual_seq_len():
    """
    Feature: test npu_fusion_attention rejects TND SP without actual_seq_len.
    Description: TND with sequence parallel requires actual_seq_qlen and actual_seq_kvlen.
    Expectation: Raise ValueError.
    """
    master_port = 11034
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_tnd_sp_without_actual_seq_len"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_bnsd_kv_seq_sharding_blocked():
    """
    Feature: test npu_fusion_attention blocks KV sequence sharding for BNSD.
    Description: Both Q and KV sharded on seq dim in BNSD layout.
    Expectation: Raise NotImplementedError.
    """
    master_port = 11035
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_bnsd_kv_seq_sharding_blocked"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_bsh_custom_scale():
    """
    Feature: test npu_fusion_attention with custom scale value.
    Description: BSH DP with scale=0.125 instead of default.
    Expectation: Output matches standalone execution with same scale.
    """
    master_port = 11036
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_custom_scale"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_dropout():
    """
    Feature: test npu_fusion_attention with dropout.
    Description: BSH DP with keep_prob=0.9, smoke test due to randomness.
    Expectation: Correct output shape without crash.
    """
    master_port = 11037
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_dropout"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_long_sequence_sp():
    """
    Feature: test npu_fusion_attention SP with long sequence.
    Description: BSH SP with seq_len=2048.
    Expectation: Correct output shape.
    """
    master_port = 11038
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_long_sequence_sp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_large_batch_dp():
    """
    Feature: test npu_fusion_attention DP with large batch.
    Description: BSH DP with batch_size=64.
    Expectation: Correct output shape.
    """
    master_port = 11039
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_large_batch_dp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_bsh_redistribute_then_attention():
    """
    Feature: test npu_fusion_attention after redistribute.
    Description: Redistribute from Replicate to Shard before running attention.
    Expectation: Output matches standalone execution.
    """
    master_port = 11040
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bsh_redistribute_then_attention"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_sp_sparse_mode_2_with_2way_split():
    """
    Feature: test npu_fusion_attention SP sparse_mode=2 with 2-way split.
    Description: Verify LEFT_UP_TO_RIGHT_DOWN offset on (2, 4) mesh.
    Expectation: Output matches standalone execution.
    """
    master_port = 11041
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_sp_sparse_mode_2_with_2way_split"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sp_sparse_mode_3_with_2way_split():
    """
    Feature: test npu_fusion_attention SP sparse_mode=3 with 2-way split.
    Description: Verify RIGHT_DOWN_TO_RIGHT_DOWN offset on (2, 4) mesh.
    Expectation: Output matches standalone execution.
    """
    master_port = 11042
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_sp_sparse_mode_3_with_2way_split"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_bnsd_sp_correctness():
    """
    Feature: test npu_fusion_attention BNSD SP correctness.
    Description: BNSD layout with Q seq shard, KV replicate, compare with standalone.
    Expectation: Output matches standalone execution.
    """
    master_port = 11043
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_bnsd_sp_correctness"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp_correctness():
    """
    Feature: test npu_fusion_attention TND DP correctness with actual_seq_len adjustment.
    Description: TND layout with 8-way DP on T dimension, verifies pure DP correctly
        adjusts actual_seq_qlen/actual_seq_kvlen via clamp while skipping sparse params
        adjustment. Gather output and compare with standalone.
    Expectation: Output matches standalone execution.
    """
    master_port = 11044
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_tnd_dp_correctness"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_tnd_cp():
    """
    Feature: test npu_fusion_attention TND layout with context parallelism.
    Description: Q Shard(0) on sp, KV Replicate, sparse_mode=3 required by CP.
        actual_seq_qlen/kvlen adjusted via clamp logic for s1_split_num > 1.
    Expectation: Output matches standalone execution.
    """
    master_port = 11045
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_tnd_cp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fa_vs_sdpa_cross_validation():
    """
    Feature: Cross-validate npu_fusion_attention (FA) against SDPA as ground truth.
    Description: Run the same Q/K/V through both PyTorch native SDPA and NPU FA (BSH layout),
        then compare outputs to detect FA kernel-level precision issues that self-comparison
        tests cannot catch.
    Expectation: FA output matches SDPA output within FP16 tolerance (atol=1e-2, rtol=1e-2).
    """
    master_port = 11046
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_fa_vs_sdpa_cross_validation"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fa_vs_sdpa_distributed_cross_validation():
    """
    Feature: Cross-validate FA against SDPA under distributed sharding.
    Description: Run the same Q/K/V through both FA (BSH) and SDPA (BNSD) with identical
        SP+MP sharding on a (4,2) mesh, gather results and compare. Verifies that FA's
        distributed sparse params adjustment and head_num adjustment produce correct results
        compared to SDPA under the same parallelism strategy.
    Expectation: FA gathered output matches SDPA gathered output within FP16 tolerance
        (atol=1e-3, rtol=1e-3).
    """
    master_port = 11047
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_fa_vs_sdpa_distributed_cross_validation"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_batch_sharding_mismatch():
    """
    Feature: test npu_fusion_attention rejects mismatched Q/K batch sharding.
    Description: BSH layout with Q Shard(0) on batch but K Replicate on batch.
    Expectation: Raise ValueError.
    """
    master_port = 11048
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_batch_sharding_mismatch"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_hidden_sharding_mismatch():
    """
    Feature: test npu_fusion_attention rejects mismatched Q/K hidden sharding.
    Description: BSH layout with Q Shard(2) on hidden but K Replicate on hidden.
    Expectation: Raise ValueError.
    """
    master_port = 11049
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_hidden_sharding_mismatch"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_dim_sharding_mismatch():
    """
    Feature: test npu_fusion_attention rejects mismatched Q/K dim sharding.
    Description: BNSD layout with Q Shard(3) on dim but K Replicate on dim.
    Expectation: Raise ValueError.
    """
    master_port = 11050
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_dim_sharding_mismatch"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_tnd_cp_wrong_sparse_mode():
    """
    Feature: test npu_fusion_attention rejects TND CP with wrong sparse_mode.
    Description: TND layout with context parallelism requires sparse_mode=3,
        but sparse_mode=0 is provided.
    Expectation: Raise ValueError.
    """
    master_port = 11051
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_tnd_cp_wrong_sparse_mode"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_error_tnd_cp_qk_global_t_mismatch():
    """
    Feature: test npu_fusion_attention rejects TND CP with mismatched global T.
    Description: TND layout with context parallelism requires Q and K to have
        the same global T-dimension.
    Expectation: Raise ValueError.
    """
    master_port = 11052
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_error_tnd_cp_qk_global_t_mismatch"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_tnd_dp_kv_sharded():
    """
    Feature: test npu_fusion_attention TND DP with kv_is_sharded=True.
    Description: TND layout with Q/K/V all Shard(0) on T dimension, triggers
        kv_is_sharded clamp branch in actual_seq_len adjustment.
    Expectation: Output matches standalone execution.
    """
    master_port = 11053
    file_name = "parallel_op_flash_attention_score.py"
    case_name = "test_tnd_dp_kv_sharded"
    torchrun_case(file_name, case_name, master_port)
