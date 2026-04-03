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
"""launch _test_fully_shard_ignore_params.py cases"""
from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case

file_name = "_test_fully_shard_replicate_params.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard_replicate_params():
    """
    Feature: Test_zero3_fully_shard.
    Description: Test_zero3_fully_shard with 1D FSDP mesh.
    Expectation: case run successfully.
    """
    master_port = 12343
    case_name = "test_zero3_fully_shard_replicate_params"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard_replicate_params_with_offload():
    """
    Feature: test_zero3_fully_shard_replicate_params_with_offload.
    Description: Test_zero3_fully_shard with 1D FSDP mesh and offload.
    Expectation: case run successfully.
    """
    master_port = 12343
    case_name = "test_zero3_fully_shard_replicate_params_with_offload"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard_replicate_params():
    """
    Feature: test_zero3_partial_shard_replicate_params.
    Description: test_zero3_partial_shard_replicate_params with 2D HSDP mesh.
    Expectation: case run successfully.
    """
    master_port = 12344
    case_name = "test_zero3_partial_shard_replicate_params"
    torchrun_case(file_name, case_name, master_port)
