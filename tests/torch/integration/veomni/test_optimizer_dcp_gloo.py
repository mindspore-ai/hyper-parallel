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
"""Framework-free launcher for VeOmni Muon DCP provenance regression."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@pytest.mark.skipif(
    find_spec("veomni") is None,
    reason="Optional VeOmni integration dependency is required",
)
@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_optimizer_dcp_gloo(tmp_path, monkeypatch):
    """Verify actual checkpoint provenance with VeOmni's partial-load planner."""
    monkeypatch.setenv("HYPER_MUON_DCP_TEST_OUTPUT", str(tmp_path))
    worker = Path(__file__).with_name("_test_optimizer_dcp_gloo.py")
    torchrun_case(str(worker), "test_optimizer_dcp_gloo", num_proc=2)
