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
"""Launcher for the opt-in MindIO worker-replacement acceptance test."""

import os

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level2", card_mark="allcards",
          essential_mark="unessential")
def test_worker_replacement():
    """Use a separate worker file so collection does not import training frameworks."""
    if os.getenv("HP_RUN_MINDIO_RECOVERY_ST") != "1":
        pytest.skip("set HP_RUN_MINDIO_RECOVERY_ST=1 with MindIO and idle NPUs")
    worker = os.path.join(os.path.dirname(__file__), "_test_process_recovery.py")
    torchrun_case(worker, "test_worker_replacement", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level2", card_mark="allcards",
          essential_mark="unessential")
def test_text_trainer_replacement():
    """Verify the complete Trainer through the public test launcher.

    Feature: Process recovery of the complete text training lifecycle.
    Description: Replace configured workers while using real offline data.
    Expectation: All ranks resume and match uninterrupted training exactly.
    """
    if os.getenv("HP_RUN_TEXT_TRAINER_RECOVERY_ST") != "1":
        pytest.skip("set HP_RUN_TEXT_TRAINER_RECOVERY_ST=1 with MindIO, TaskD and idle NPUs")
    worker = os.path.join(os.path.dirname(__file__), "_test_process_recovery.py")
    torchrun_case(worker, "test_text_trainer_replacement", num_proc=1)
