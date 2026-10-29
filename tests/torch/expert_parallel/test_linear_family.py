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
"""Device-free launcher for Dense Family A5 training."""

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_linear_family() -> None:
    """
    Feature: Scheme-selected Dense Linear training.
    Description: Compare baseline and MXFP8 replacements with their existing implementations.
    Expectation: Ten training steps match outputs, gradients and updates exactly.
    """
    torchrun_case("tests/torch/expert_parallel/_test_linear_family.py", "test_linear_family", num_proc=1)
