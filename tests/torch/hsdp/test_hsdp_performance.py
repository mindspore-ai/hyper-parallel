# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test hsdp performance feature with torchrun 8 card"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

HSDP_PREFETCH = "hsdp_prefetch.py"
HSDP_COMM_ASYNC = "hsdp_comm_async.py"
HSDP_COMM_FUSION = "hsdp_comm_fusion.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_forward_prefetch():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_hsdp_forward_prefetch
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP_PREFETCH, "test_hsdp_forward_prefetch", 12341, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_backward_prefetch():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_hsdp_backward_prefetch
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP_PREFETCH, "test_hsdp_backward_prefetch", 12342, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_comm_async():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_hsdp_comm_async
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP_COMM_ASYNC, "test_hsdp_comm_async", 12343, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_comm_fusion():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_hsdp_comm_fusion
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP_COMM_FUSION, "test_hsdp_comm_fusion", 12344, 8)
    ])
