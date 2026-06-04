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
"""Pytest launchers for FSDP + Expert Parallelism distributed NPU precision tests.

Tests verify FSDP + EP composition correctness:
  - FE-01: Forward + backward numerical correctness (FSDP 2-card × EP 2-card)
  - FE-03: 3-step training convergence (overfit on small data)

Port allocation:
  10500  FSDP + EP forward/backward (4 cards: 2 FSDP × 2 EP)
  10502  FSDP + EP 3-step training (4 cards: 2 FSDP × 2 EP)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(Path(__file__).resolve().parent / "_test_fsdp_ep_distributed.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``.

    Args:
        *cases: Each element is a tuple ``(case_name, master_port, num_proc)``.
    """
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_fsdp_ep_forward_backward():
    """
    Feature: FSDP + EP forward and backward precision on NPU.
    Description:
        4-card setup: 2 FSDP ranks × 2 EP ranks.  MoE has 4 experts total,
        each FSDP rank holds a full copy of all 4 experts (replicated on FSDP dim),
        and each EP rank holds 2 experts (sharded on EP dim).  Verifies that
        FSDP+EP output and input gradients match the standalone reference within
        rtol=1e-3, atol=1e-3.
    Expectation: Run success.
    """
    _run_group(
        ("test_fsdp_ep_forward_backward_npu", 10500, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_fsdp_ep_three_step_training():
    """
    Feature: FSDP + EP 3-step training convergence.
    Description:
        4-card setup: 2 FSDP × 2 EP.  Train for 3 steps on the same small batch.
        Loss should decrease monotonically (overfit verification).
    Expectation: Run success with loss decreasing.
    """
    _run_group(
        ("test_fsdp_ep_three_step_training_npu", 10502, 4),
    )
