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
"""Thin launchers for two-rank reentrant checkpoint/FSDP integration."""

import os
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(Path(__file__).with_name("reentrant_composite_cases.py").resolve())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_fsdp_checkpoint_exclude() -> None:
    """Compare wrapper and explicit API gradients in disjoint two-rank groups."""
    cases = [
        TorchCase(_WORKER, "test_fsdp_checkpoint_wrapper_exclude", num_proc=2),
        TorchCase(_WORKER, "test_fsdp_checkpoint_api_exclude", num_proc=2),
    ]
    budget = int(os.environ.get("HYPER_PARALLEL_TEST_CARD_COUNT", "8"))
    if budget < 2:
        pytest.skip("FSDP checkpoint exclusion requires two NPU cards")
    if budget > 8:
        raise ValueError("HYPER_PARALLEL_TEST_CARD_COUNT must be at most 8")
    for offset in range(0, len(cases), budget // 2):
        parallel_run(cases[offset:offset + budget // 2], global_num_proc=budget)
