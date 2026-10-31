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
"""Two framework-free launchers for the bounded MoonEP/MegaMoE contract suite."""
import os
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port
from tests.torch.multicore._case_launcher import run_multicore_case
from tests.torch.multicore._test_env import prepare_multicore_test_environment, without_inherited_rank_environment


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
def test_moonep_contracts(monkeypatch: pytest.MonkeyPatch, dispatch_mode: str) -> None:
    """Cover representative precision and lifetime combinations with one worker group per transport."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", "2")
    monkeypatch.setenv("HP_MEGA_MOE_DISPATCH_MODE", dispatch_mode)
    monkeypatch.setenv("PYTEST_ADDOPTS", (os.getenv("PYTEST_ADDOPTS", "") + " -x").strip())
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(str(Path(__file__).with_name("_test_mega_moe_moonep.py")),
                           "test_moonep_contracts", num_proc=2)
