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
"""Thin launcher for native HP VLM Dataset integration."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_native_vlm_dp2_gloo() -> None:
    """Feature: Native HP VLM sample balancing.
    Description: Route real decoded images through native sampling, collation, prefetch and checkpoint replay.
    Expectation: Membership, labels and image ownership survive; deterministic DP toy gradients match native batches.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_vlm_gloo.py")),
        case_name="test_native_vlm_dp2_gloo", num_proc=2,
    )
