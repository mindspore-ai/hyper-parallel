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
"""test hsdp performancee feature with msrun 8 card"""
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.hsdp.hsdp_test_common import run_hsdp_case_by_name


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_forward_prefetch():
    """
    Feature: hsdp prefetch
    Description: test hsdp forward prefetch
    Expectation: run success
    """
    file_name = "hsdp_prefetch.py"
    case_name = "test_hsdp_forward_prefetch"
    run_hsdp_case_by_name(file_name, case_name)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_backward_prefetch():
    """
    Feature: hsdp prefetch
    Description: test hsdp backward prefetch
    Expectation: run success
    """
    file_name = "hsdp_prefetch.py"
    case_name = "test_hsdp_backward_prefetch"
    run_hsdp_case_by_name(file_name, case_name)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_comm_async():
    """
    Feature: hsdp enable async comm
    Description: test hsdp async comm performance
    Expectation: run success
    """
    file_name = "hsdp_comm_async.py"
    case_name = "test_hsdp_comm_async"
    run_hsdp_case_by_name(file_name, case_name)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hsdp_comm_fusion():
    """
    Feature: hsdp enable comm fusion
    Description: test hsdp comm fusion performance
    Expectation: run success
    """
    file_name = "hsdp_comm_fusion.py"
    case_name = "test_hsdp_comm_fusion"
    run_hsdp_case_by_name(file_name, case_name)
