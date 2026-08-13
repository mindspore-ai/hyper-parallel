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
"""Launcher for finalized-layout pretrained weight loading on NPU."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case

_WORKER = str(Path(__file__).resolve().parent / "_test_pretrained_weight_loading.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pretrained_weights_follow_final_layout_npu() -> None:
    """
    Feature: Two-card pretrained weight loading.
    Description: Launch the finalized-layout loader worker through torchrun.
    Expectation: Both ranks load the correct DTensor and _sharding_spec shards.
    """
    torchrun_case(
        _WORKER,
        "test_pretrained_weights_follow_final_layout_npu",
        master_port=13871,
        num_proc=2,
    )
