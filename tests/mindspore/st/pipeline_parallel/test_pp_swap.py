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
"""msrun launcher for MindSpore GPipe activation swap tests."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_pp_swap.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_gpipe_pipeline_swap():
    """
    Feature: GPipe activation swap.
    Description: Run a two-rank GPipe schedule with activation swap enabled.
    Expectation: Distributed output and local gradients match the serial reference.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE, "test_gpipe_pipeline_swap", worker_num=2, local_worker_num=2),
    ])
