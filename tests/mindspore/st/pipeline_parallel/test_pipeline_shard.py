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
"""msrun entry for MindSpore pipeline_parallel/pipeline_shard."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.distributed_launcher import msrun_case

_FILE_NAME = os.path.join(os.path.dirname(__file__), "_pipeline_shard.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_pipeline_shard():
    """
    Feature: PipelineParallel + shard.
    Description: Run the MindSpore interleaved pipeline + shard case and compare with standalone.
    Expectation: Run success.
    """
    msrun_case(2, _FILE_NAME, "test_pipeline_shard", 18611, worker_num=4, local_worker_num=4)
