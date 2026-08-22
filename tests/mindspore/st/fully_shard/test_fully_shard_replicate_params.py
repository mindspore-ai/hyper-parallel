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
"""Launch MindSpore fully_shard replicate_params ST."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_fully_shard_replicate_params.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_ms_fully_shard_replicate_params():
    """
    Feature: fully_shard replicate_params precision and ignored TP DTensor state (MindSpore).
    Description: Run the replicate-weights/sharded-biases precision case (with prefetch lifecycle check)
                 and the ignored TP-sharded DTensor state case together in one 8-card wave.
    Expectation: Replicate grads match the full reference, sharded grads match the shard,
                 and ignored state stays visible.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE, "test_ms_fully_shard_with_replicate_params",
                      worker_num=4, local_worker_num=4),
        MindSporeCase(_TEST_FILE, "test_ms_fully_shard_ignored_dtensor_state",
                      worker_num=4, local_worker_num=4),
    ])
