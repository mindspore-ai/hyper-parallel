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
"""Pytest launchers for the Llama3 parallel-vs-single-card accuracy suite.

Each launcher uses :func:`tests.common.parallel_case.parallel_run` to spawn the
worker cases in ``_test_llama3_accuracy.py`` on disjoint NPU subsets. The
worker compares per-step training loss between a parallel scenario (TP+FSDP or
TP+CP+FSDP) and an in-process single-card baseline computed from the same
random seed and optimizer state.
"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase
from tests.common.distributed_launcher import torchrun_case

_WORKER = "_test_llama3_accuracy.py"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
)
def test_llama3_single_card_baseline():
    """
    Feature: Llama3 single-card baseline runs as a sanity check in the accuracy suite.
    Description: Launch the single-card worker with one rank and verify the deterministic
        single-card loss trajectory contains :data:`_STEPS` finite values.
    Expectation: Worker exits with status 0 and prints the reference loss trajectory.
    """
    torchrun_case(_WORKER, "test_single_card_baseline", master_port=12601, num_proc=1)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_llama3_tp_fully_shard_accuracy():
    """
    Feature: ``TP + fully_shard`` Llama3 accuracy vs single-card.
    Description:
        Launch :func:`test_tp_fully_shard_matches_single_card` on a 4-rank ``(dp=2, tp=2)`` mesh
        and assert that each step's reconstructed global loss matches the in-process single-card
        reference within ``rtol=1e-3``/``atol=1e-3``.
    Expectation: Worker exits with status 0 on every rank.
    """
    parallel_run([
        TorchCase(_WORKER, "test_tp_fully_shard_matches_single_card", 12602, 4),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_llama3_tp_cp_fully_shard_accuracy():
    """
    Feature: ``TP + CP + fully_shard`` Llama3 accuracy vs single-card.
    Description:
        Launch :func:`test_tp_cp_fully_shard_matches_single_card` on an 8-rank
        ``(dp=2, cp=2, tp=2)`` mesh with Colossal-style context parallel attached to every BSHD
        SDPA core, and assert that each step's reconstructed global loss matches the in-process
        single-card reference within ``rtol=1e-3``/``atol=1e-3``.
    Expectation: Worker exits with status 0 on every rank.
    """
    torchrun_case(_WORKER, "test_tp_cp_fully_shard_matches_single_card", master_port=12603)
