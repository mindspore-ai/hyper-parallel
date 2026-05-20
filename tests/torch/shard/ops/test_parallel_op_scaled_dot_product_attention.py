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
"""test parallel op scaled dot product attention"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_SDPA = "parallel_op_scaled_dot_product_attention.py"

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_replicate
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_replicate", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group1_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_sdpa_replicate
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_replicate", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_dp
        2.test_sdpa_mp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_dp", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_mp", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group2_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_sdpa_dp
        2.test_sdpa_mp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_dp", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_mp", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_sp
        2.test_sdpa_dp_mp_2d
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_dp_mp_2d", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group3_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_sdpa_sp
        2.test_sdpa_dp_mp_2d
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_dp_mp_2d", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group4():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_sp_mp_2d
        2.test_sdpa_sp_causal
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_mp_2d", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_causal", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group4_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_sdpa_sp_mp_2d
        2.test_sdpa_sp_causal
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_mp_2d", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_causal", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group5():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_sp_explicit_mask
        2.test_sdpa_error_kv_strategy_mismatch
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_explicit_mask", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_error_kv_strategy_mismatch", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group5_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_sdpa_sp_explicit_mask
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_explicit_mask", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_scaled_dot_product_attention_group6():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_custom_scale
        2.test_sdpa_dropout
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_custom_scale", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_dropout", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group6_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_sdpa_custom_scale
        2.test_sdpa_dropout
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_custom_scale", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_dropout", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_parallel_op_scaled_dot_product_attention_group7():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_enable_gqa
        2.test_sdpa_sp_correctness
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_enable_gqa", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_correctness", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group7_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_sdpa_enable_gqa
        2.test_sdpa_sp_correctness
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_enable_gqa", num_proc=4),
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_sp_correctness", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scaled_dot_product_attention_group8():
    """
    Feature: parallel run case in shard
    Description:
        1.test_sdpa_error_kv_seq_sharding
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SDPA, "test_sdpa_error_kv_seq_sharding", num_proc=4),
    ])
