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

"""Lightweight EP8 launcher for the MoonEP collective numerical gate."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_moonep_collective_fp32():
    """Launch complete FP32 gradient/update parity on eight NPU ranks."""
    worker = Path(__file__).with_name("_moonep_collective_parity.py")
    torchrun_case(str(worker), "test_collective_fp32", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_moonep_collective_bf16_mixed_sgd():
    """Launch BF16 tokens plus FP32 route weights with the eager expert path."""
    worker = Path(__file__).with_name("_moonep_collective_parity.py")
    torchrun_case(str(worker), "test_collective_bf16_mixed_sgd", num_proc=8)
