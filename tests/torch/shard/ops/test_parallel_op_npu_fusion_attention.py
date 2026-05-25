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
"""Test runner for npu_fusion_attention distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_npu_fusion_attention.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_npu_fusion_attention_group1():
    """
    Feature: parallel run case in _test_parallel_op_npu_fusion_attention
    Description:
        1. test_bsh_replicate — BSH format replicated
        2. test_bsh_dp — BSH format data parallel
        3. test_bsh_mp — BSH format model parallel
        4. test_bsh_sp — BSH format sequence parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_bsh_replicate", num_proc=2),
        TorchCase(IMPL_FILE, "test_bsh_dp", num_proc=2),
        TorchCase(IMPL_FILE, "test_bsh_mp", num_proc=2),
        TorchCase(IMPL_FILE, "test_bsh_sp", num_proc=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_npu_fusion_attention_group2():
    """
    Feature: parallel run case in _test_parallel_op_npu_fusion_attention
    Description:
        1. test_bsh_dp_mp_2d — BSH format 2D data+model parallel
        2. test_bsh_sp_mp_2d — BSH format 2D sequence+model parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_bsh_dp_mp_2d", num_proc=4),
        TorchCase(IMPL_FILE, "test_bsh_sp_mp_2d", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_npu_fusion_attention_group3():
    """
    Feature: parallel run case in _test_parallel_op_npu_fusion_attention
    Description:
        1. test_bsh_dp_sp_mp_3d — BSH format 3D mesh parallel
        2. test_fa_vs_sdpa_distributed_cross_validation — FA vs SDPA distributed cross validation
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_bsh_dp_sp_mp_3d", num_proc=4),
        TorchCase(IMPL_FILE, "test_fa_vs_sdpa_distributed_cross_validation", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_npu_fusion_attention_group4():
    """
    Feature: parallel run case in _test_parallel_op_npu_fusion_attention
    Description:
        1. test_sp_sparse_mode_0 — SP sparse mode 0
        2. test_sp_sparse_mode_2 — SP sparse mode 2
        3. test_sp_sparse_mode_3 — SP sparse mode 3
        4. test_sp_sparse_mode_4 — SP sparse mode 4
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_sp_sparse_mode_0", num_proc=2),
        TorchCase(IMPL_FILE, "test_sp_sparse_mode_2", num_proc=2),
        TorchCase(IMPL_FILE, "test_sp_sparse_mode_3", num_proc=2),
        TorchCase(IMPL_FILE, "test_sp_sparse_mode_4", num_proc=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_npu_fusion_attention_group5():
    """
    Feature: parallel run case in _test_parallel_op_npu_fusion_attention
    Description:
        1. test_dp_sparse_mode_1 — DP sparse mode 1
        2. test_dp_sparse_mode_4 — DP sparse mode 4
        3. test_bsh_custom_scale — BSH format custom scale
        4. test_bsh_redistribute_then_attention — BSH redistribute then attention
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_dp_sparse_mode_1", num_proc=2),
        TorchCase(IMPL_FILE, "test_dp_sparse_mode_4", num_proc=2),
        TorchCase(IMPL_FILE, "test_bsh_custom_scale", num_proc=2),
        TorchCase(IMPL_FILE, "test_bsh_redistribute_then_attention", num_proc=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_npu_fusion_attention_group6():
    """
    Feature: parallel run case in _test_parallel_op_npu_fusion_attention
    Description:
        1. test_sp_sparse_mode_2_with_2way_split — SP sparse mode 2 with 2-way split
        2. test_sp_sparse_mode_3_with_2way_split — SP sparse mode 3 with 2-way split
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_sp_sparse_mode_2_with_2way_split", num_proc=4),
        TorchCase(IMPL_FILE, "test_sp_sparse_mode_3_with_2way_split", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_npu_fusion_attention_group7():
    """
    Feature: parallel run case in _test_parallel_op_npu_fusion_attention
    Description:
        1. test_bnsd_sp_correctness — BNSD format SP correctness
        2. test_tnd_dp_correctness — TND format DP correctness
        3. test_tnd_cp — TND format CP
        4. test_tnd_dp_kv_sharded — TND format DP with KV sharded
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_bnsd_sp_correctness", num_proc=2),
        TorchCase(IMPL_FILE, "test_tnd_dp_correctness", num_proc=2),
        TorchCase(IMPL_FILE, "test_tnd_cp", num_proc=2),
        TorchCase(IMPL_FILE, "test_tnd_dp_kv_sharded", num_proc=2),
    ])
