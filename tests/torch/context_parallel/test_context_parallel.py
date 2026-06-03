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
"""Pytest launchers for ContextParallel NPU tests.

Each launcher uses ``parallel_run`` to execute several worker cases from
``_test_context_parallel.py`` concurrently on disjoint NPU subsets.

Typical packing:
  - up to four 2-card worker cases per launcher
  - up to two 4-card worker cases per launcher
  - one 8-card worker case per launcher
"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = "_test_context_parallel.py"


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_context_parallel_group1():
    """
    Feature: parallel_run launcher for merged 2-card Ulysses coverage
    Description:
        1. test_ulysses_bnsd_suite
        2. test_ulysses_bsnd_fa_noncausal
        3. test_ulysses_tnd_fa_noncausal
        4. test_colossal_bsh_fa_noncausal
    Expectation: Run success.
    """
    _run_group(
        ("test_ulysses_bnsd_suite", 13000, 2),
        ("test_ulysses_bsnd_fa_noncausal", 13003, 2),
        ("test_ulysses_tnd_fa_noncausal", 13006, 2),
        ("test_colossal_bsh_fa_noncausal", 13010, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_context_parallel_group2():
    """
    Feature: parallel_run launcher for merged 2-card Colossal layout coverage
    Description:
        1. test_colossal_sbh_fa_noncausal
        2. test_colossal_bsnd_fa_noncausal
        3. test_colossal_bsnd_fa_causal_leftup
        4. test_colossal_bnsd_suite
    Expectation: Run success.
    """
    _run_group(
        ("test_colossal_sbh_fa_noncausal", 13011, 2),
        ("test_colossal_bsnd_fa_noncausal", 13012, 2),
        ("test_colossal_bsnd_fa_causal_leftup", 13013, 2),
        ("test_colossal_bnsd_suite", 13014, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_context_parallel_group3():
    """
    Feature: parallel_run launcher for 2-card TND, load-balance, and API coverage
    Description:
        1. test_colossal_tnd_fa_causal
        2. test_lb_bsnd_fa_causal
        3. test_parallelize_module_api_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_colossal_tnd_fa_causal", 13020, 2),
        ("test_lb_bsnd_fa_causal", 13030, 2),
        ("test_parallelize_module_api_npu", 13040, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_context_parallel_group4():
    """
    Feature: parallel_run launcher for heavier 2-card load-balance and async coverage
    Description:
        1. test_lb_bnsd_fa_causal
        2. test_async_ulysses_suite
        3. test_async_colossal_suite
        4. test_async_vs_sync_precision_suite
    Expectation: Run success.
    """
    _run_group(
        ("test_lb_bnsd_fa_causal", 13031, 2),
        ("test_async_ulysses_suite", 13200, 2),
        ("test_async_colossal_suite", 13201, 2),
        ("test_async_vs_sync_precision_suite", 13203, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_context_parallel_group5():
    """
    Feature: parallel_run launcher for merged 4-card Hybrid BNSD coverage
    Description:
        1. test_hybrid_bnsd_suite
        2. test_hybrid_tnd_fa_causal
    Expectation: Run success.
    """
    _run_group(
        ("test_hybrid_bnsd_suite", 13100, 4),
        ("test_hybrid_tnd_fa_causal", 13103, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_context_parallel_group6():
    """
    Feature: parallel_run launcher for 4-card Hybrid mesh and async coverage
    Description:
        1. test_hybrid_2d_mesh_equiv
        2. test_async_hybrid_forward_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_hybrid_2d_mesh_equiv", 13104, 4),
        ("test_async_hybrid_forward_npu", 13202, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_context_parallel_group7():
    """
    Feature: parallel_run launcher for 4-card TP+CP integration coverage
    Description:
        1. test_tp_cp_combination_npu
        2. test_tp_cp_dtensor_composed_mesh_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_tp_cp_combination_npu", 13110, 4),
        ("test_tp_cp_dtensor_composed_mesh_npu", 13120, 4),
    )
