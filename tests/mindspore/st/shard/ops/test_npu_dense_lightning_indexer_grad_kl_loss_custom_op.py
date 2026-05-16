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
"""Test runner for npu_dense_lightning_indexer_grad_kl_loss custom op ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(
    Path(__file__).resolve().parent / "npu_dense_lightning_indexer_grad_kl_loss_custom_op.py"
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_grad_kl_loss_group1():
    """
    Feature: parallel run case in npu_dense_lightning_indexer_grad_kl_loss_custom_op
             (BSND replicated, DP, dp+cp)
    Description:
        1. test_grad_kl_loss_bsnd_replicated — BSND all replicated, fwd+bwd vs standalone
        2. test_grad_kl_loss_bsnd_dp — BSND B-dim data parallel, fwd+bwd vs standalone
        3. test_grad_kl_loss_bsnd_dp_cp — 4-card 2-D mesh (dp=2, cp=2); fwd+bwd
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_grad_kl_loss_bsnd_replicated", 20021, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_grad_kl_loss_bsnd_dp", 20022, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_grad_kl_loss_bsnd_dp_cp", 20023, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_grad_kl_loss_group2():
    """
    Feature: parallel run case in npu_dense_lightning_indexer_grad_kl_loss_custom_op
             (TND replicated, DP, dp+cp)
    Description:
        1. test_grad_kl_loss_tnd_replicated — TND all replicated, fwd+bwd vs standalone
        2. test_grad_kl_loss_tnd_dp — TND T1-dim data parallel, fwd+bwd vs standalone
        3. test_grad_kl_loss_tnd_dp_cp — 4-card 2-D mesh TND dp+cp, fwd+bwd
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_grad_kl_loss_tnd_replicated", 20025, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_grad_kl_loss_tnd_dp", 20026, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_grad_kl_loss_tnd_dp_cp", 20027, 4, 4, 2),
    ])
