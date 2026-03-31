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
"""Test fully_shard precision comparison with standalone baseline"""
import inspect
import os
import shutil
import subprocess

import pytest
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case

# Temporary directory for baseline artifacts
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_baseline")
file_name = "_precision_fully_shard.py"


def run_baseline():
    """Run standalone baseline training"""
    print("Running standalone baseline training...")
    cmd = f"python {os.path.dirname(__file__)}/_precision_baseline.py"
    process = subprocess.run(cmd, shell=True, check=False)

    assert process.returncode == 0, "Baseline training failed"

    print("Baseline training completed successfully")


def run_case(case_name):
    """Run the specified test case with setup and teardown"""
    try:
        if os.path.exists(TEMP_DIR):
            print(f"Cleaning up existing temporary directory before start: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)

        run_baseline()

        glob_v = 2
        master_port = 18178
        msrun_case(glob_v, file_name, case_name, master_port)
    finally:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print(f"Cleaned up temporary directory: {TEMP_DIR}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard():
    """
    Feature: Compare fully_shard precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_partial_shard():
    """
    Feature: Compare partial_shard precision with standalone baseline
    Description: Run standalone baseline and partial_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@pytest.mark.skip(reason="Case failed after upgrading mindspore version")
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_replicate_params():
    """
    Feature: Compare partial_shard precision with standalone baseline
    Description: Run standalone baseline and partial_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_grad_accum():
    """
    Feature: Compare fully_shard gradient accumulation precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with gradient accumulation,
                 then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_empty_weight():
    """
    Feature: Compare empty init fully_shard precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")
