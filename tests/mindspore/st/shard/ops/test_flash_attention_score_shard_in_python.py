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
"""parallel_flash_attention_score_shard_in_python test"""
from tests.common.parallel_case import parallel_run, MindSporeCase
from tests.common.mark_utils import arg_mark

FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON = "flash_attention_score_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_flash_attention_score_shard_in_python_group1():
    """
    Feature: parallel run case in flash_attention_score_shard_in_python
    Description:
        1. test_flash_attention_score_model_parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(FLASH_ATTENTION_SCORE_SHARD_IN_PYTHON, "test_flash_attention_score_model_parallel", 18289, 4, 4)
    ])
