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
"""test vpp"""
import os
from tests.common.mark_utils import arg_mark


def run_case(case_name, master_port):
    cmd = f"torchrun --nproc-per-node=8 " \
          f"--master_addr=127.0.0.1 --master_port={master_port} " \
          f"-m pytest -s pipeline_shard.py::{case_name}"
    ret = os.system(cmd)
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_simple_mlp():
    """
    Feature: schedule vpp + shard.
    Description: Test pp.
    Expectation: Run success.
    """
    case_name = "test_vpp_shard"
    master_port = 12346
    run_case(case_name, master_port)
