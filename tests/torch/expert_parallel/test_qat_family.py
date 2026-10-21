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
"""Device-free launcher for the QAT Family A5 cases."""

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_qat_family() -> None:
    """
    Feature: QAT selection through Module Family.
    Description: Compare Family and direct strategy training for named W8/native W4/fake W4 schemes.
    Expectation: Ten steps preserve outputs, gradients and updates exactly.
    """
    torchrun_case("tests/torch/expert_parallel/_test_qat_family.py", "test_qat_family", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scheme_fused_mxfp8() -> None:
    """
    Feature: Named MXFP8 scheme with optional fusion.
    Description: Compare routed training against the existing flag-selected fused implementation.
    Expectation: Ten steps match exactly and execute real fused kernels.
    """
    torchrun_case("tests/torch/expert_parallel/_test_qat_family.py", "test_scheme_fused_mxfp8", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_deepseek_qat_family() -> None:
    """
    Feature: Non-fused V4.1 expert training through Module Family.
    Description: Native clipped SwiGLU and the existing W8/native W4/fake W4 strategies form the reference.
    Expectation: Ten steps match outputs, input/weight gradients and updates exactly, including clipping.
    """
    torchrun_case("tests/torch/expert_parallel/_test_deepseek_qat_family.py", "test_deepseek_qat_family", num_proc=1)
