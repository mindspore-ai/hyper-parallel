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
"""test hyper_parallel.shard with different type of args"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

SHARD_API_ARGS = "shard_api_args.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_shard_api_args_group1():
    """
    Feature: parallel run case in shard_api_args
    Description:
        1. test_shard_with_args_and_kwargs_non_dtensor_input
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SHARD_API_ARGS, "test_shard_with_args_and_kwargs_non_dtensor_input", 18308, 4, 4),
    ])
