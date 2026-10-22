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
"""Thin single-NPU launcher for GroupedExperts MXFP8 acceptance."""

import importlib

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_family_mxfp8_local():
    """
    Feature: Local GroupedExperts MXFP8 Family.
    Description: Exercise real fused/non-fused training, physical saves and numerical references.
    Expectation: Both implementations satisfy fixed tolerances and repeated-backward contracts.
    """
    worker = importlib.import_module("tests.torch.expert_parallel._test_family_mxfp8_local")
    worker.test_fused_boundary()
    worker.test_fused_offload_checkpoint()
    worker.test_single_device(False)
    worker.test_single_device(True)
    worker.test_fused_comparison()
    worker.test_three_way_loss()
