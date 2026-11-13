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
"""Explicit single-NPU launch for native unified attention numerical acceptance."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_unified_attention(tmp_path: Path) -> None:
    """Run native operator backward and tiny HP/HF optimizer steps on an explicitly selected NPU."""
    device = os.environ.get("RL_UNIFIED_ATTENTION_ST_DEVICE")
    if device is None:
        pytest.skip("Set RL_UNIFIED_ATTENTION_ST_DEVICE to an available logical NPU index")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_unified_attention.py")), "--device", device,
        "--output", str(tmp_path / "attention.json")], check=True,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_unified_attention_backend(tmp_path: Path) -> None:
    """Verify real paged-cache writes and updated-metadata graph replay against FP64."""
    device = os.environ.get("RL_UNIFIED_ATTENTION_ST_DEVICE")
    model = os.environ.get("RL_UNIFIED_ST_MODEL")
    if device is None or not model:
        pytest.skip("Set RL_UNIFIED_ATTENTION_ST_DEVICE and RL_UNIFIED_ST_MODEL for the NPU backend test")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_unified_attention_backend.py")),
         "--device", device, "--model", model, "--output", str(tmp_path / "backend.json")], check=True,
    )
