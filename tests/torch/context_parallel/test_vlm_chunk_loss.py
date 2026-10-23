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
"""Lightweight CPU launcher for VLM chunk loss with both CP implementations."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_vlm_chunk_loss() -> None:
    """Feature: VLM chunk loss with context parallelism.

    Description: Compare chunk/full-logits execution on two Gloo ranks for both CP algorithms.
    Expectation: Global losses and all parameter gradients match within numerical tolerance.
    """
    torchrun_case(str(Path(__file__).with_name("_test_vlm_chunk_loss.py")), "test_chunk_loss_cp", num_proc=2)
