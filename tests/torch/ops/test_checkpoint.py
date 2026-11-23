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
"""Lightweight single-card launcher for operator selection during NPU checkpointing."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_checkpoint.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_checkpoint_backend_selection() -> None:
    """Verify native and fused NPU operator outputs and gradients under checkpointing."""
    torchrun_case(file_name=_WORKER, case_name="test_checkpoint_backend_selection", num_proc=1)
