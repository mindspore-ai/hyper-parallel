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
"""test activation checkpoint"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

ACTIVATION_CHECKPOINT = "activation_checkpoint.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ac_memory_comparison():
    """
    Feature: parallel run case in activation_checkpoint
    Description:
        1. test_ac_memory_comparison
        2. test_checkpoint_wrapper_accepts_func
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(ACTIVATION_CHECKPOINT, "test_ac_memory_comparison", 12404, 1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_checkpoint_wrapper_accepts_func", 12405, 1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_overlap_wrapper", 12406, 1)
    ])
