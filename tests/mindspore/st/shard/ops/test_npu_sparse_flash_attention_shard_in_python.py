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
"""Test runner for npu_sparse_flash_attention distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(
    Path(__file__).resolve().parent / "npu_sparse_flash_attention_shard_in_python.py"
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_sfa_ms_group1():
    """
    Feature: parallel run case in npu_sparse_flash_attention_shard_in_python (BSND+TND replicated
             and DP, 8 cards)
    Description:
        1. test_sfa_bsnd_replicated — BSND all replicated, fwd+bwd vs standalone
        2. test_sfa_bsnd_dp — BSND B-dim data parallel, fwd+bwd vs standalone
        3. test_sfa_tnd_replicated — TND all replicated, fwd+bwd vs standalone
        4. test_sfa_tnd_dp — TND T1-dim data parallel (q AND k Shard(0)); local seq_lens
           per rank; q_split==k_split so no CP seq_len adjustment.
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_sfa_bsnd_replicated", 19200, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_sfa_bsnd_dp", 19201, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_sfa_tnd_replicated", 19203, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_sfa_tnd_dp", 19202, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_sfa_ms_group2():
    """
    Feature: parallel run case in npu_sparse_flash_attention_shard_in_python (BSND+TND CP, 4 cards)
    Description:
        1. test_sfa_bsnd_cp — BSND, q/si/q_rope Shard(1) on S1, k/v/k_rope Replicate;
           validates TP-as-CP pattern (MindFormers: S1 sharded by cp_tp combined)
        2. test_sfa_tnd_cp — TND, q/si/q_rope Shard(0) on T1, k/v/k_rope Replicate;
           seq_lens adjusted per rank by _tnd_cp_impl
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_sfa_bsnd_cp", 19205, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_sfa_tnd_cp", 19204, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_sfa_ms_group3():
    """
    Feature: parallel run case in npu_sparse_flash_attention_shard_in_python (BSND+TND dp+cp
             2-D mesh, 8 cards)
    Description:
        1. test_sfa_bsnd_dp_cp — 4-card 2-D mesh (dp=2, cp=2); B sharded by dp, S1 sharded
           by cp; k/v/key_rope replicated on cp. Mirrors MindFormers BSND shard() spec.
        2. test_sfa_tnd_dp_cp — 4-card 2-D mesh (dp=2, cp=2); T1 of q/si/q_rope sharded
           by dp AND cp (4-way split); T2 of k/v/k_rope sharded by dp only (2-way split).
           Mirrors MindFormers TND shard() spec with dp_cp_tp combined sequence split.
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_sfa_bsnd_dp_cp", 19206, 4, 4, 2),
        MindSporeCase(IMPL_FILE, "test_sfa_tnd_dp_cp", 19207, 4, 4, 2),
    ])
