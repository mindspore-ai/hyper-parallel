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
"""Launcher for shared Gated DeltaNet context parallel tests."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(Path(__file__).resolve().parent / "_test_linear_attention_context_parallel.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_linear_attention_context_parallel_accuracy():
    """
    Feature: Gated DeltaNet context parallel execution
    Description: Compare Ulysses, P2P, and AllGather forward/backward with a full-sequence reference.
    Expectation: Outputs, input gradients, parameter gradients, and gradient norms match.
    """
    parallel_run(
        [
            TorchCase(
                _WORKER,
                "test_linear_attention_cp_forward_backward_accuracy",
                num_proc=2,
            )
        ]
    )
