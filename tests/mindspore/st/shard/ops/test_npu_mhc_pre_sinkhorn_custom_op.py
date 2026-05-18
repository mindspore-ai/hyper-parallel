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
"""Test runner for npu_mhc_pre_sinkhorn distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(
    Path(__file__).resolve().parent / "npu_mhc_pre_sinkhorn_custom_op.py"
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mhc_pre_sinkhorn_group1():
    """
    Feature: parallel run case in npu_mhc_pre_sinkhorn_custom_op (BSND 2-card distributed).
    Description:
        1. test_mhc_pre_sinkhorn_bsnd_replicated — BSND all replicated on 2 cards
        2. test_mhc_pre_sinkhorn_bsnd_dp — BSND B-dim data parallel on 2 cards
        Note: test_mhc_pre_sinkhorn_tnd_dp_cp_tp is temporarily commented out.
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mhc_pre_sinkhorn_bsnd_replicated", 20051, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_mhc_pre_sinkhorn_bsnd_dp", 20052, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mhc_pre_sinkhorn_group2():
    """
    Feature: parallel run case in npu_mhc_pre_sinkhorn_custom_op (BSND 8-card dp_cp_tp).
    Description:
        1. test_mhc_pre_sinkhorn_bsnd_dp_cp_tp — BSND dp=2, cp=2, tp=2 on 8 cards
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mhc_pre_sinkhorn_bsnd_dp_cp_tp", 20054, 8, 8, 2),
    ])
