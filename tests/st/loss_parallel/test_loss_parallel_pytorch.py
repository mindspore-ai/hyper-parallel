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
"""Entry file for PyTorch loss_parallel accuracy tests.

This file launches distributed tests using torchrun.
"""
import os
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


DIST_FILE = os.path.join(os.path.dirname(__file__), "_loss_parallel_pytorch_dist.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_loss_parallel_pytorch_accuracy():
    """
    Feature: loss_parallel accuracy test (PyTorch backend)
    Description:
        1. test_single_vs_multi_card_loss_parity
        2. test_loss_parallel_context_correctness
        3. test_gradient_correctness_with_loss_parallel
    Expectation: All tests pass with numerical precision within tolerance.
    """
    parallel_run([
        TorchCase(DIST_FILE, "test_single_vs_multi_card_loss_parity", 30001, 2),
        TorchCase(DIST_FILE, "test_loss_parallel_context_correctness", 30002, 2),
        TorchCase(DIST_FILE, "test_gradient_correctness_with_loss_parallel", 30003, 2),
    ])
