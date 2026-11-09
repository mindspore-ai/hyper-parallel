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
"""Thin launcher for four-card buffered NPU data loading."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_buffered_device_loading_hccl() -> None:
    """Feature: Buffered accelerator data loading.
    Description: Exercise DP2/TP2 native loading and DP4 sources with asynchronous H2D.
    Expectation: All tensor fields reach the device and sample membership survives collective transport.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_buffered_device_loading_hccl.py")),
        case_name="test_buffered_device_loading_hccl", num_proc=4,
    )
