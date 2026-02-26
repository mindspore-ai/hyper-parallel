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
"""Test scaled_dot_product_attention distributed operator"""

from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_sdpa_replicate():
    """
    Feature: test scaled_dot_product_attention with no parallelism.
    Description: All tensors replicated, verify basic DTensor pass-through.
    Expectation: Output matches standalone execution.
    """
    master_port = 11100
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_replicate"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_dp():
    """
    Feature: test scaled_dot_product_attention with data parallelism.
    Description: Shard on batch dimension (dim 0).
    Expectation: Output matches standalone execution.
    """
    master_port = 11101
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_dp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_mp():
    """
    Feature: test scaled_dot_product_attention with head parallelism.
    Description: Shard on head dimension (dim 1).
    Expectation: Output matches standalone execution.
    """
    master_port = 11102
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_mp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_sp():
    """
    Feature: test scaled_dot_product_attention with sequence parallelism.
    Description: Q shard on seq dim (dim 2), KV replicated.
    Expectation: Correct local output shape.
    """
    master_port = 11103
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_sp"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_dp_mp_2d():
    """
    Feature: test scaled_dot_product_attention with DP + MP on 2D mesh.
    Description: Shard batch and head on (4, 2) mesh.
    Expectation: Output matches standalone execution.
    """
    master_port = 11104
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_dp_mp_2d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_sp_mp_2d():
    """
    Feature: test scaled_dot_product_attention with SP + MP on 2D mesh.
    Description: Q shard on seq + head, KV replicate seq + shard head.
    Expectation: Correct local output shape.
    """
    master_port = 11105
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_sp_mp_2d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_sp_causal():
    """
    Feature: test scaled_dot_product_attention SP with is_causal=True.
    Description: Causal mask constructed per Q chunk for sequence parallel.
    Expectation: Output matches standalone causal execution.
    """
    master_port = 11106
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_sp_causal"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_sp_explicit_mask():
    """
    Feature: test scaled_dot_product_attention SP with explicit attn_mask.
    Description: Global attn_mask sliced to local Q range for sequence parallel.
    Expectation: Output matches standalone execution.
    """
    master_port = 11107
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_sp_explicit_mask"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_error_kv_strategy_mismatch():
    """
    Feature: test scaled_dot_product_attention rejects mismatched KV strategies.
    Description: Key and Value have different sharding strategies.
    Expectation: Raise ValueError.
    """
    master_port = 11108
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_error_kv_strategy_mismatch"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sdpa_error_kv_seq_sharding():
    """
    Feature: test scaled_dot_product_attention rejects KV sequence sharding.
    Description: Both Q and KV sharded on seq dim without Ring Attention.
    Expectation: Raise NotImplementedError.
    """
    master_port = 11109
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_error_kv_seq_sharding"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sdpa_custom_scale():
    """
    Feature: test scaled_dot_product_attention with custom scale.
    Description: DP with explicit scale=0.125.
    Expectation: Output matches standalone execution with same scale.
    """
    master_port = 11110
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_custom_scale"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sdpa_dropout():
    """
    Feature: test scaled_dot_product_attention with dropout.
    Description: DP with dropout_p=0.1, smoke test due to randomness.
    Expectation: Correct output shape without crash.
    """
    master_port = 11111
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_dropout"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sdpa_enable_gqa():
    """
    Feature: test scaled_dot_product_attention with GQA enabled.
    Description: Q has more heads than KV, enable_gqa=True with DP.
    Expectation: Correct output shape.
    """
    master_port = 11112
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_enable_gqa"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_sdpa_sp_correctness():
    """
    Feature: test scaled_dot_product_attention SP correctness.
    Description: SP with Q shard, KV replicate, gather and compare with standalone.
    Expectation: Output matches standalone execution.
    """
    master_port = 11113
    file_name = "parallel_op_scaled_dot_product_attention.py"
    case_name = "test_sdpa_sp_correctness"
    torchrun_case(file_name, case_name, master_port)
