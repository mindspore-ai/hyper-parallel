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
"""Launch MegaGate tests through the actual DeepSeek V4.1 adapter."""

from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.torch.multicore._test_env import (
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).with_name("_test_mega_gate_deepseek_v41.py"))
_HCCL_PORT_RANGE = "62100-62163"


@pytest.fixture(scope="module")
def deepseek_v41_environment() -> None:
    """Fail before spawning NPU workers when the model dependency is unavailable."""
    result = subprocess.run(
        [sys.executable, "-c", (
            "from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config; "
            "from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer"
        )],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        pytest.fail(
            f"DeepSeek V4.1 model dependencies are unavailable in {sys.executable}. "
            "Use the model training environment with Transformers DeepSeek-V4 support "
            "(the V4.1 adapter targets Transformers 5.13), then rerun with python -m pytest. "
            f"Dependency import failed:\n{result.stderr}",
            pytrace=False,
        )


@pytest.mark.usefixtures("deepseek_v41_environment")
@pytest.mark.parametrize("case,num_proc", [
    ("test_router_parity", 1),
    ("test_moe_parity", 1),
    ("test_fsdp_bf16_parity", 2),
    ("test_ep_parity", 2),
])
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_mega_gate_deepseek_v41(case: str, num_proc: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the native router and model-owned distributed execution paths."""
    prepare_multicore_test_environment()
    if num_proc > 1:
        monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", _HCCL_PORT_RANGE)
    with without_inherited_rank_environment():
        parallel_run([TorchCase(_WORKER, case, num_proc=num_proc)], global_num_proc=num_proc)
