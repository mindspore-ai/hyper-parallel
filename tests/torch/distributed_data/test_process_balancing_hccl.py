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
"""Thin launchers for process data communication and nested workers."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_native_process_hccl() -> None:
    """Feature: Independent native data processes.
    Description: Run DP2/CP2 with nested workers, a nonzero planner and checkpoint restore.
    Expectation: Reads and packing run outside training; resumed input matches the saved suffix.
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_test_process_balancing_hccl.py")),
                  case_name="test_native_process_hccl", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_external_process_hccl() -> None:
    """Feature: Independent external data processes.
    Description: Run nested DataLoader workers, child HCCL and parent H2D on DP4.
    Expectation: Accepted balancing preserves global occurrences and epoch replay.
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_test_process_balancing_hccl.py")),
                  case_name="test_external_process_hccl", num_proc=4)
