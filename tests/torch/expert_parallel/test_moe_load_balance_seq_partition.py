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
"""Pytest launchers for sequence_partition_group distributed tests.

Each test function uses ``parallel_run`` to spawn ``torchrun`` workers that
execute the actual distributed logic in
``_test_moe_load_balance_seq_partition.py``.

Port allocation:
  10510  LB-S03: 4-card sequence_partition_group — expert_fraction sync matches manual
  10512  LB-S04: Global normalization changes loss
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(
    Path(__file__).resolve().parent / "_test_moe_load_balance_seq_partition.py"
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
def test_lbs03_expert_fraction_sync_matches_manual():
    """
    Feature: MoE load balance with sequence_partition_group.
    Description:
        4-card gloo. Each rank has different routing data.
        With sequence_partition_group, expert_fraction is all-reduced
        and loss matches manual global computation.
    Expectation: Run success.
    """
    _run_group(
        ("test_lbs03_expert_fraction_sync_matches_manual", 10510, 4),
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_lbs04_global_normalization_changes_loss():
    """
    Feature: MoE load balance — global normalization changes loss.
    Description:
        Without sequence_partition_group, loss uses local token counts.
        With the group, loss uses global normalization (num_sub_sequence),
        producing a different (smaller) loss value.
    Expectation: Run success.
    """
    _run_group(
        ("test_lbs04_global_normalization_changes_loss", 10512, 4),
    )
