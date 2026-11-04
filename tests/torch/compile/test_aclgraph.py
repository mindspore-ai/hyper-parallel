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
"""Launch native ACLGraph numerical and distributed replay checks."""

from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode,world_size,inductor_mode", [
    ("static", 1, "off"), ("static", 2, "off"), ("dynamic", 2, "off"),
    ("static", 1, "regional"), ("static", 2, "regional"), ("dynamic", 2, "regional"),
])
def test_aclgraph_training(mode, world_size, inductor_mode):
    """Compare loss, accumulated gradients and optimizer updates with eager."""
    worker = str(Path(__file__).with_name("_test_aclgraph.py"))
    torchrun_case(file_name=worker, case_name=f"test_aclgraph_training[{mode}-{inductor_mode}]", num_proc=world_size)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_aclgraph_mutated_state():
    """Check that recording/replay updates live state exactly once per call."""
    worker = str(Path(__file__).with_name("_test_aclgraph.py"))
    torchrun_case(file_name=worker, case_name="test_aclgraph_mutated_state", num_proc=1)
