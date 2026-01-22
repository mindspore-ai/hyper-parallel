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
"""test fully_shard api"""

from tests.torch.utils import torchrun_case

def test_fully_shard_01():
    """
    Feature: Test fully_shard with simple network, optimization level is default ZeRO-3
    Description: The DenseNet only have one weight and no bias, verify the basic process of fully_shard
    Expectation: run successfully
    """
    master_port = 12342
    file_name = "_test_fully_shard.py"
    case_name = "test_fully_shard_01"
    torchrun_case(file_name, case_name, master_port)


def test_fully_shard_02():
    """
    Feature: Test fully_shard with multi-layer network, optimization level is default ZeRO-3
    Description: The FullyShardTestNet is a multi-layer module, verify the basic process of fully_shard
    Expectation: run successfully
    """
    master_port = 12342
    file_name = "_test_fully_shard.py"
    case_name = "test_fully_shard_02"
    torchrun_case(file_name, case_name, master_port)
