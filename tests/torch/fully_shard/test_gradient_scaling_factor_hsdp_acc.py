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
"""Launch _test_gradient_scaling_factor_hsdp_acc.py distributed case."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_FILE = "_test_gradient_scaling_factor_hsdp_acc.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gradient_scaling_factor_hsdp_acc_group1():
    """
    Feature: parallel run case for HSDPModule.set_gradient_scaling_factor under
        HSDP + gradient accumulation + replicate_params (pure all-reduce path).
    Description: 4-card HSDP (2x2) mesh; verify the scaling factor scales every
        parameter's accumulated gradient exactly once, including replicate params.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FILE, "test_gradient_scaling_factor_hsdp_grad_accumulation", 12383, 4),
    ])
