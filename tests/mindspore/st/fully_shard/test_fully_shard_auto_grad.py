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
"""launch _test_fully_shard_auto_grad.py cases (MindSpore)"""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_fully_shard_auto_grad.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_chunked_output_fully_shard():
    """
    Feature: fully_shard autograd with chunked output (MindSpore)
    Description: Verify that a fully_shard-wrapped OutputLayer can be called
        multiple times in a for-loop with results concatenated and a single backward.
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE, "test_chunked_output_fully_shard", worker_num=2, local_worker_num=2),
    ])
