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
"""Tests for init_empty_weights -> fully_shard -> init weight consistency."""

from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_init_weights_consistency():
    """
    Feature: init_empty_weights -> fully_shard -> load weights, single/multi-card consistency
    Description: Verify that weights loaded after init_empty_weights + fully_shard match
                 the single-card reference when gathered across all ranks.
    Expectation: run successfully
    """
    master_port = 12350
    file_name = "_test_init_weights.py"
    case_name = "test_init_weights_consistency"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_init_weights_with_randn_like():
    """
    Feature: init_empty_weights -> fully_shard -> non-in-place random ops
    Description: Verify non-in-place random ops (randn_like, rand_like)
                 dispatch correctly on DTensors and buffers.
    Expectation: run successfully
    """
    master_port = 12351
    file_name = "_test_init_weights.py"
    case_name = "test_init_weights_with_randn_like"
    torchrun_case(file_name, case_name, master_port)
