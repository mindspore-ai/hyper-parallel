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
"""Opt-in NPU graph parity acceptance; no framework import during collection."""

import os
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.distributed_launcher import torchrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("ranks", [2, 4])
def test_moe_graph(ranks: int, tmp_path: Path) -> None:
    """Compare old/new eager and replay with real uneven and skewed routes."""
    if os.environ.get("RL_MOE_GRAPH_ST") != "1":
        pytest.skip("Set RL_MOE_GRAPH_ST=1 after reserving Ascend NPUs")
    previous = os.environ.get("RL_MOE_GRAPH_OUTPUT")
    os.environ["RL_MOE_GRAPH_OUTPUT"] = str(tmp_path)
    try:
        torchrun_case(str(Path(__file__).with_name("_moe_graph.py")), "test_worker", num_proc=ranks)
    finally:
        if previous is None:
            os.environ.pop("RL_MOE_GRAPH_OUTPUT", None)
        else:
            os.environ["RL_MOE_GRAPH_OUTPUT"] = previous
