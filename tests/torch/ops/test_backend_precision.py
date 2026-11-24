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
"""Lightweight single-card launcher for registered backend/native precision parity."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_backend_precision.py"))


@arg_mark(plat_marks=["platform_ascend910b", "platform_gpu"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_backend_precision() -> None:
    """Compare actual registered accelerator kernels with same-device native, including VJPs."""
    torchrun_case(file_name=_WORKER, case_name="test_backend_precision", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_native_precision() -> None:
    """Validate same-device native baselines against independent formulas on Ascend."""
    torchrun_case(file_name=_WORKER, case_name="test_native_precision", num_proc=1)
