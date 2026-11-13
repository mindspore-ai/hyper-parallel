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
"""Explicit opt-in launcher for the optional Ascend SHMEM component."""

import os
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@pytest.mark.skipif(os.environ.get("KDA_ONESIDED_TEST") != "1", reason="Requires a built A2 SHMEM component")
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_kda_onesided_core():
    """Compare complete KDA core gradients and asynchronous graph lifetimes."""
    worker = Path(__file__).with_name("_test_kda_onesided.py")
    torchrun_case(str(worker), "test_kda_onesided_core", num_proc=int(os.environ.get("KDA_ONESIDED_CP", "4")))
