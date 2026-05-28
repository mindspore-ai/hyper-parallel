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
"""Pytest launchers for MoE load balance distributed sync tests.

Each test function uses ``parallel_run`` to spawn ``torchrun`` workers that
execute the actual distributed logic in ``_test_moe_load_balance_distributed.py``.

Port allocation:
  10500  LB-D01: 4-card DP sync — expert_bias identical after sync
  10502  LB-D02: Compare with/without all_reduce
  10504  LB-D03: Pure EP (no dp_group) — independent update
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(
    Path(__file__).resolve().parent / "_test_moe_load_balance_distributed.py"
)


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``.

    Args:
        *cases: Each element is a tuple ``(case_name, master_port, num_proc)``.
    """
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_lbd01_dp_sync_expert_bias_identical():
    """
    Feature: MoE load balance distributed sync with DP group.
    Description:
        4-card gloo DP. Each rank has different input → different tokens_per_expert.
        After sync_and_update_expert_bias with dp_group, all ranks produce
        identical expert_bias.
    Expectation: Run success.
    """
    _run_group(
        ("test_lbd01_dp_sync_expert_bias_identical", 10500, 4),
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_lbd02_sync_vs_nosync():
    """
    Feature: MoE load balance — sync vs no-sync bias comparison.
    Description:
        Without dp_group, expert_bias diverges across ranks (different inputs).
        With dp_group, all ranks produce identical expert_bias.
    Expectation: Run success.
    """
    _run_group(
        ("test_lbd02_sync_vs_nosync", 10502, 4),
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_lbd03_pure_ep_no_dp_group():
    """
    Feature: MoE load balance — pure EP without dp_group.
    Description:
        Pure EP scenario (all ranks see same routing data). No dp_group needed.
        sync_and_update_expert_bias without any group should work correctly.
    Expectation: Run success.
    """
    _run_group(
        ("test_lbd03_pure_ep_no_dp_group", 10504, 4),
    )
