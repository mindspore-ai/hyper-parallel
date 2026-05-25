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
"""Test runner for all_gather_matmul distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(
    Path(__file__).resolve().parent / "_test_parallel_op_all_gather_matmul.py"
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_agm_ms_group1():
    """
    Feature: parallel run cases in all_gather_matmul_shard_in_python (2-card cases, 8 cards total)
    Description:
        1. test_agm_tp_input_shard0_x2_replicate — x1 Shard(0) on tp, x2 Replicate, gather_output=True
        2. test_agm_tp_input_shard0_x2_shard1   — x1 Shard(0), x2 Shard(1) on n-dim
        3. test_agm_trans_x2_true               — trans_x2=True, x2 physical (N,K) Shard(0) on n
        4. test_agm_gather_output_false          — gather_output=False; verify gather_out is empty
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_agm_tp_input_shard0_x2_replicate", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_agm_tp_input_shard0_x2_shard1", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_agm_trans_x2_true", worker_num=2, local_worker_num=2, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_agm_gather_output_false", worker_num=2, local_worker_num=2, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_agm_ms_group2():
    """
    Feature: parallel run cases in all_gather_matmul_shard_in_python (4-card cases)
    Description:
        1. test_agm_dp_tp_input_shard0    — 2D (dp=2, tp=2) mesh, x1 Shard(0) on tp; fwd+grad
        2. test_agm_k_sharded_mp2_tp2     — 2D (mp=2, tp=2) mesh, m on mp + k on tp; Partial output
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_agm_dp_tp_input_shard0", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_agm_k_sharded_mp2_tp2", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_agm_ms_group3():
    """
    Feature: parallel run cases in all_gather_matmul_shard_in_python (8-card case)
    Description:
        1. test_agm_k_sharded_mp2_np2_tp2 — 3D (mp=2, np=2, tp=2) mesh, m/n/k sharded on distinct axes
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_agm_k_sharded_mp2_np2_tp2", worker_num=8, local_worker_num=8, glog_v=2),
    ])
