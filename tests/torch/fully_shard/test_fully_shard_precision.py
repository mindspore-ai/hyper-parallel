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
"""launch _test_fully_shard_precison.py cases"""
from tests.torch.utils import torchrun_case

file_name = "_test_fully_shard_precision.py"
def test_zero3_fully_shard():
    """
    Feature: Test_zero3_fully_shard.
    Description: Test_zero3_fully_shard with 1D FSDP mesh.
    Expectation: case run successfully.
    """
    master_port = 12343
    case_name = "test_zero3_fully_shard"
    torchrun_case(file_name, case_name, master_port)


def test_zero3_partial_shard():
    """
    Feature: Test_zero3_fully_shard.
    Description: Test_zero3_fully_shard with 2D HSDP mesh.
    Expectation: case run successfully.
    """
    master_port = 12344
    case_name = "test_zero3_partial_shard"
    torchrun_case(file_name, case_name, master_port)