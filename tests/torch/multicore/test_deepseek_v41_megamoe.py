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

"""Lightweight launchers for DSV4.1 dynamic experts and hybrid parallelism."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.torch.multicore._integration import EXTENDED_LEVEL1, run_acceptance_worker

_PARALLEL_CASES = [(2, 1, False, "push"), (2, 1, True, "pull"),
                   (1, 2, True, "push"), (1, 2, False, "pull"),
                   (2, 2, False, "push"), (2, 2, True, "pull")]
if EXTENDED_LEVEL1:
    _PARALLEL_CASES = [(tp, cp, vision, mode) for tp, cp in ((2, 1), (1, 2), (2, 2))
                       for vision in (False, True) for mode in ("push", "pull")]


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_deepseek_v41_megamoe_single_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Check routed/shared outputs and gradients through the V4.1 MegaMoe adapter."""
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    run_acceptance_worker(monkeypatch, tmp_path, "test_deepseek_v41_single_card", 1,
                          "_test_deepseek_v41_parallel.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_deepseek_v41_megamoe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str) -> None:
    """Run EP8/E48 forward and all-gradient comparisons at limit 10."""
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    run_acceptance_worker(monkeypatch, tmp_path, "test_deepseek_v41_dynamic_tokens", 8,
                           "_test_deepseek_v41_dynamic_tokens.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
@pytest.mark.parametrize("tp,cp,vision,dispatch_mode", _PARALLEL_CASES)
def test_deepseek_v41_megamoe_parallel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_mode: str,
                                     tp: int, cp: int, vision: bool) -> None:
    """Run real TP/SP+EP and CP+EP boundaries, router, shared experts, and gradients."""
    monkeypatch.setenv("HP_DSV41_TP", str(tp))
    monkeypatch.setenv("HP_DSV41_CP", str(cp))
    monkeypatch.setenv("HP_DSV41_VISION", str(int(vision)))
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    run_acceptance_worker(monkeypatch, tmp_path, "test_deepseek_v41_parallel", 8,
                           "_test_deepseek_v41_parallel.py", variant=f"tp{tp}-cp{cp}-vision{int(vision)}")
