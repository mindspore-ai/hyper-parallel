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
"""Test runner for matmul_reduce_scatter distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(
    Path(__file__).resolve().parent / "matmul_reduce_scatter_shard_in_python.py"
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mrs_ms_group1():
    """
    Feature: parallel run cases in matmul_reduce_scatter_shard_in_python (2-card cases, 6 cards total)
    Description:
        1. test_mrs_tp_basic       — x1 Shard(1) on k, x2 Shard(0) on k; reference = matmul + ReduceScatter
        2. test_mrs_trans_x2_true  — trans_x2=True, x2 physical (N,K) Shard(1) on k
        3. test_mrs_large_m        — larger M dimension (M=256)
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mrs_tp_basic",      20200, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_mrs_trans_x2_true", 20201, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_mrs_large_m",       20202, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mrs_ms_group2():
    """
    Feature: parallel run cases in matmul_reduce_scatter_shard_in_python (4-card case)
    Description:
        1. test_mrs_dp_tp_basic — 2D (dp=2, tp=2) mesh, x1 Shard(1) on tp, x2 Shard(0) on tp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mrs_dp_tp_basic", 20203, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mrs_ms_group3():
    """
    Feature: parallel run cases in matmul_reduce_scatter_shard_in_python (8-card case)
    Description:
        1. test_mrs_mp2_np2_tp2 — 3D (mp=2, np=2, tp=2) mesh, m/n/k sharded on distinct axes
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mrs_mp2_np2_tp2", 20204, 8, 8, 2),
    ])
