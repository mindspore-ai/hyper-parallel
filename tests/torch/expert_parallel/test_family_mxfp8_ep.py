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
"""Thin two-NPU launcher for MXFP8 Module Family expert parallelism."""

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_family_mxfp8_ep() -> None:
    """
    Feature: MXFP8 Module Family with EP=2.
    Description: Build planner shards and train fused/non-fused paths with sparse routing.
    Expectation: Outputs and gradients match an ordered reference, including an empty receiver.
    """
    torchrun_case(file_name=__file__.replace("test_family_", "_test_family_"),
                  case_name="test_family_mxfp8_ep", num_proc=2)
