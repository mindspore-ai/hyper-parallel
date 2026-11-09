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
"""Thin launcher for window prefetch with concurrent model communication."""

from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("accumulation,device_depth", ((1, 1), (3, 0), (3, 1), (3, 2), (5, 2)))
def test_optimizer_step_loading_hccl(accumulation: int, device_depth: int) -> None:
    """Feature: Window double buffering on NPU.
    Description: Prepare next-window inputs while DP2/TP2 ranks compute and issue model-group collectives.
    Expectation: H2D, A2A, TP broadcasts and complete-step checkpoint replay preserve exact samples.

    Args:
        accumulation: Micro-batches prepared per optimizer step.
        device_depth: Future micro-batches staged on Device.
    """
    # Separate workers prevent successive configurations from wrapping NPU stream pools.
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_optimizer_step_hccl.py")),
        case_name=f"test_optimizer_step_loading_hccl[{accumulation}-{device_depth}]", num_proc=4,
    )
