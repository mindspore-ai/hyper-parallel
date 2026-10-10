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
"""Thin launcher for Host Engram sparse owner precision cases."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).resolve().parent / "_test_host_engram_sparse.py")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_host_engram_four_rank_gloo():
    """Check the four-rank Host owner step.

    Feature: EP2 Host Engram with two same-owner replicas.
    Description: Compare device lookups, sparse gradients, clipping, and restore.
    Expectation: Every row, norm, and checkpoint invariant matches the fixture.
    """
    torchrun_case(_WORKER, "test_host_engram_four_rank_gloo", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_host_engram_four_rank_hccl():
    """Check the four-rank owner step on Ascend.

    Feature: Ascend HCCL Host Engram routing and update.
    Description: Run the same device-versus-Host precision fixture on NPUs.
    Expectation: Outputs, gradients, and restored rows match the reference.
    """
    torchrun_case(_WORKER, "test_host_engram_four_rank_hccl", num_proc=4)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_host_engram_ep1_generalization_gloo():
    """Check EP1 loss-domain generalization.

    Feature: EP1 with DP2, CP2, and TP2 loss-parallel variants.
    Description: Compare Host lookup and gradients to a device embedding.
    Expectation: Each topology has the expected normalized rows and norm.
    """
    torchrun_case(_WORKER, "test_host_engram_ep1_generalization_gloo", num_proc=2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_host_engram_pp_empty_stage_gloo():
    """Check a PP stage without an Engram table.

    Feature: PP2 WORLD norm and empty-stage checkpoint participation.
    Description: Put the Host table on one stage and dense weights on both.
    Expectation: Both stages agree on norm and restore completes.
    """
    torchrun_case(_WORKER, "test_host_engram_pp_empty_stage_gloo", num_proc=2)
