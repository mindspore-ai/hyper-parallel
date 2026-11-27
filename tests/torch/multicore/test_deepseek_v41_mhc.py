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
"""Thin launcher for public and model-owned fused mHC precision checks."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.torch.multicore._integration import activate_multicore_environment


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_fused_mhc_native_parity(monkeypatch):
    """Compare both fused entry points against the V4.1 CPU semantic reference."""
    activate_multicore_environment(monkeypatch)
    worker = str(Path(__file__).with_name("_test_deepseek_v41_mhc.py"))
    torchrun_case(worker, "test_fused_mhc_native_parity", num_proc=1)
