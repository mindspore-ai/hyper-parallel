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
import os
import shutil
import subprocess

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

# Temporary directory for baseline artifacts
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_baseline")
_PRECISION_IMPL = os.path.join(os.path.dirname(__file__), "_fully_shard_precision.py")
_LIST_PRECISION_IMPL = os.path.join(os.path.dirname(__file__), "_precision_fully_shard_list.py")


def _generate_baseline() -> None:
    """Generate shared baseline artifacts (standalone checkpoint + loss reference)."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    print("Running standalone baseline training...")
    cmd = f"python {_PRECISION_IMPL} --generate-baseline"
    process = subprocess.run(cmd, shell=True, check=False)
    assert process.returncode == 0, "Baseline training failed"
    print("Baseline training completed successfully")


def _cleanup_baseline() -> None:
    """Remove baseline artifacts directory if it exists."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned up temporary directory: {TEMP_DIR}")


# ---------------------------------------------------------------------------
# 2-card groups: baseline shared within the group
# ---------------------------------------------------------------------------

def _run_basic_group(case_names):
    """Generate baseline once, then run the given 2-card precision cases via parallel_run."""
    try:
        _generate_baseline()
        parallel_run([
            MindSporeCase(_PRECISION_IMPL, name, worker_num=2, local_worker_num=2)
            for name in case_names
        ])
    finally:
        _cleanup_baseline()


def _run_comm_fusion_group(case_names):
    """Generate baseline once, then run comm_fusion 2-card cases."""
    try:
        _generate_baseline()
        parallel_run([
            MindSporeCase(_PRECISION_IMPL, name, worker_num=2, local_worker_num=2)
            for name in case_names
        ])
    finally:
        _cleanup_baseline()


def _run_grad_accum_group(case_names):
    """Generate baseline once, then run grad_accum 2-card cases."""
    try:
        _generate_baseline()
        parallel_run([
            MindSporeCase(_PRECISION_IMPL, name, worker_num=2, local_worker_num=2)
            for name in case_names
        ])
    finally:
        _cleanup_baseline()


# ---------------------------------------------------------------------------
# 4-card 2D-mesh group
# ---------------------------------------------------------------------------

def _run_partial_shard_group(case_names):
    """Generate baseline once, then run partial_shard 4-card cases."""
    try:
        _generate_baseline()
        parallel_run([
            MindSporeCase(_PRECISION_IMPL, name, worker_num=4, local_worker_num=4)
            for name in case_names
        ])
    finally:
        _cleanup_baseline()


# ---------------------------------------------------------------------------
# list-unit precision group (no shared baseline; each uses get_group_size())
# ---------------------------------------------------------------------------

def _run_list_precision_group(case_names):
    """Run list-unit precision cases (no shared baseline; each uses get_group_size())."""
    try:
        _cleanup_baseline()
        parallel_run([
            MindSporeCase(_LIST_PRECISION_IMPL, name, worker_num=4, local_worker_num=4)
            for name in case_names
        ])
    finally:
        _cleanup_baseline()


# ============================================================================
# Test functions
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_basic_suite():
    """
    Feature: fully_shard basic and prefetch/recompute/empty_init precision.
    Description:
        1. test_ms_zero3_fully_shard — basic fully_shard with 1D dp mesh.
        2. test_ms_zero3_fully_shard_prefetch — with child-module prefetch.
        3. test_ms_zero3_fully_shard_prefetch_recompute — with prefetch + activation recompute.
        4. test_ms_zero3_fully_shard_empty_weight — with init_empty_weights.
    Expectation: All losses match standalone baseline within tolerance.
    """
    _run_basic_group([
        "test_ms_zero3_fully_shard",
        "test_ms_zero3_fully_shard_prefetch",
        "test_ms_zero3_fully_shard_prefetch_recompute",
        "test_ms_zero3_fully_shard_empty_weight",
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_comm_fusion_suite():
    """
    Feature: fully_shard comm_fusion precision.
    Description:
        1. test_ms_zero3_fully_shard_comm_fusion — with comm_fusion.
        2. test_ms_zero3_fully_shard_comm_fusion_prefetch — with comm_fusion + prefetch.
    Expectation: All losses match standalone baseline within tolerance.
    """
    _run_comm_fusion_group([
        "test_ms_zero3_fully_shard_comm_fusion",
        "test_ms_zero3_fully_shard_comm_fusion_prefetch",
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_grad_accum_suite():
    """
    Feature: fully_shard gradient accumulation (zero3 / comm_fusion) precision.
    Description:
        1. test_ms_zero3_fully_shard_grad_accum — basic gradient accumulation.
        2. test_ms_zero3_fully_shard_prefetch_recompute_grad_accum — with prefetch + recompute.
        3. test_ms_zero3_fully_shard_grad_accum_comm_fusion — with comm_fusion.
    Expectation: All losses match standalone baseline within tolerance.
    """
    _run_grad_accum_group([
        "test_ms_zero3_fully_shard_grad_accum",
        "test_ms_zero3_fully_shard_prefetch_recompute_grad_accum",
        "test_ms_zero3_fully_shard_grad_accum_comm_fusion",
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_zero3_partial_shard_suite():
    """
    Feature: fully_shard with 2D (dp, op) partial shard mesh.
    Description:
        1. test_ms_zero3_partial_shard — basic partial shard.
        2. test_ms_zero3_partial_shard_prefetch_recompute — with prefetch + recompute.
    Expectation: All losses match standalone baseline within tolerance.
    """
    _run_partial_shard_group([
        "test_ms_zero3_partial_shard",
        "test_ms_zero3_partial_shard_prefetch_recompute",
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_list_unit_precision_suite():
    """
    Feature: fully_shard(list) numerical parity vs standalone (MindSpore).
    Description:
        1. test_ms_fully_shard_list_unit_precision — fully_shard([dense1, dense2], reshard_after_forward=False);
           compare final loss and dense1 grad shard to non-sharded training.
        2. test_ms_fully_shard_list_unit_prefetch_recompute_precision — same with prefetch + recompute.
    Expectation: Run success (grouped hooks aligned with PyTorch FSDP2 list semantics).
    """
    _run_list_precision_group([
        "test_ms_fully_shard_list_unit_precision",
        "test_ms_fully_shard_list_unit_prefetch_recompute_precision",
    ])
