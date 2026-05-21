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
file_name = "_fully_shard_precision.py"
file_name_list = "_precision_fully_shard_list.py"


def run_baseline() -> None:
    """Run standalone baseline training"""
    print("Running standalone baseline training...")
    cmd = f"python {os.path.dirname(__file__)}/{file_name} --generate-baseline"
    process = subprocess.run(cmd, shell=True, check=False)

    assert process.returncode == 0, "Baseline training failed"

    print("Baseline training completed successfully")


def run_case(case_name: str) -> None:
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


def run_list_precision_case(case_name: str) -> None:
    """Run list-unit precision case (reference training lives in-script; no standalone baseline)."""
    try:
        if os.path.exists(TEMP_DIR):
            print(f"Cleaning up existing temporary directory before start: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)

        glob_v = 2
        master_port = 18525
        script_path = os.path.join(os.path.dirname(__file__), file_name_list)
        msrun_case(glob_v, script_path, case_name, master_port)
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_prefetch():
    """
    Feature: Compare fully_shard prefetch precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with prefetch enabled,
                 then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_prefetch_recompute():
    """
    Feature: Compare fully_shard prefetch + recompute precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with prefetch and activation
                 recompute enabled, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_partial_shard():
    """
    Feature: Compare partial_shard precision with standalone baseline
    Description: Run standalone baseline and partial_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_partial_shard_prefetch_recompute():
    """
    Feature: Compare partial_shard prefetch + recompute precision with standalone baseline
    Description: Run standalone baseline and partial_shard multi-card training with prefetch and activation
                 recompute enabled, then compare losses on rank 0
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
def test_ms_zero3_fully_shard_prefetch_recompute_grad_accum():
    """
    Feature: Compare fully_shard prefetch + recompute + gradient accumulation precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with prefetch, activation
                 recompute, and gradient accumulation, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@pytest.mark.skip(reason="The Mindspore framework has a bug that causes a core dump when the process exits.")
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero1_fully_shard_grad_accum():
    """
    Feature: Compare zero1-style fully_shard gradient accumulation precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training
                 on replicated parameters with gradient accumulation,
                 disable grad sync on intermediate micro steps via set_requires_gradient_sync(False),
                 and synchronize only when accumulation reaches the optimizer step
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_list_unit_precision():
    """
    Feature: fully_shard(list) numerical parity vs standalone (MindSpore).
    Description: Nested fully_shard with ``fully_shard([dense1, dense2], reshard_after_forward=False)``;
        compare final loss and dense1 grad shard to non-sharded training (same seed).
    Expectation: Run success (grouped hooks aligned with PyTorch FSDP2 list semantics).
    """
    run_list_precision_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_list_unit_prefetch_recompute_precision():
    """
    Feature: fully_shard(list) numerical parity vs standalone with prefetch and recompute (MindSpore).
    Description: Nested fully_shard with ``fully_shard([dense1, dense2], reshard_after_forward=False)``
        plus prefetch and activation recompute; compare final loss and dense1 grad shard to
        non-sharded training (same seed).
    Expectation: Run success (grouped hooks aligned with PyTorch FSDP2 list semantics).
    """
    run_list_precision_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_empty_weight():
    """
    Feature: Compare empty init fully_shard precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_comm_fusion():
    """
    Feature: Compare fully_shard comm_fusion precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with comm_fusion enabled,
                 then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_comm_fusion_prefetch():
    """
    Feature: Compare fully_shard comm_fusion + prefetch precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with comm_fusion and
                 prefetch enabled, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_fully_shard_grad_accum_comm_fusion():
    """
    Feature: Compare fully_shard gradient accumulation + comm_fusion precision with standalone baseline
    Description: Run standalone baseline and fully_shard multi-card training with gradient accumulation
                 and comm_fusion enabled, then compare losses on rank 0
    Expectation: Losses should match within tolerance
    """
    run_case(case_name=f"{inspect.stack()[0].function}")
