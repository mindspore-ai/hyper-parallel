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
"""Thin launcher for real Indexed text provider integration."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_indexed_text_dp2_tp2_gloo() -> None:
    """Feature: Indexed sample-level distributed packing.
    Description: Read real .bin/.idx files using DP2/TP2 with double buffering.
    Expectation: Plans precede payload reads and packed losses/gradients match independent documents.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_indexed_text_gloo.py")),
        case_name="test_indexed_text_dp2_tp2_gloo",
        num_proc=4,
    )
