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
"""Lightweight eight-card NPU launcher for monitor persistence acceptance."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_monitor_npu_components() -> None:
    """Run device summaries, Muon and checkpoint acceptance with shared temporary storage.

    Feature: Trainer monitor device integration.
    Description: Launch eight workers without importing Torch in the parent.
    Expectation: Every worker verifies finite summaries and exact restore state.
    """
    with tempfile.TemporaryDirectory(prefix="hp-monitor-") as directory:
        with patch.dict(os.environ, {"MONITOR_ACCEPTANCE_DIR": directory}):
            torchrun_case(
                str(Path(__file__).with_name("_test_monitor.py")),
                "test_monitor_npu_components",
                num_proc=8,
            )
