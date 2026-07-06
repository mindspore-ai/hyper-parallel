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
"""Pytest launcher for combined parallel strategy tests."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_combinations.py")

# Port allocation for this module:
#   10620-10624: 2-card tests (split into two groups to keep sum_num_proc <= 8)
#   10625-10626: 4-card group (sum = 8)
#   10627-10628: 8-card groups (each group has single 8-card case)
# These ports are chosen to avoid conflicts with other test modules.


def _run_group(*cases):
    """Launch a group of worker cases with parallel_run."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_2card_group1():
    _run_group(
        ("run_ep_only_base", 10620, 2),
        ("run_ep_only_grouped_mm", 10621, 2),
        ("run_ep_only_shared", 10622, 2),
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_2card_group2():
    _run_group(
        ("run_tp_only", 10623, 2),
        ("run_validation", 10624, 2),
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_4card_group():
    _run_group(
        ("run_dp_ep", 10625, 4),
        ("run_ep_tp", 10626, 4),
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_8card_1():
    _run_group(
        ("run_dp_ep_tp", 10627, 8),
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_8card_2():
    _run_group(
        ("run_dp_ep_cp_with_attention", 10628, 8),
    )
