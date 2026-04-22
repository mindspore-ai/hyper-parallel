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
"""Pytest launchers for ``PrepareModule*`` NPU distributed integration tests.

Worker cases live in ``_test_prepare_module_io_distributed.py``. Port allocation
(``sum(num_proc) <= 8`` per wave):

  Wave 1 (``test_prepare_module_io_two_card_wave_one``): 10600–10603, four cases × 2 ranks
  Wave 2 (``test_prepare_module_io_two_card_wave_two``): 10610–10613, four cases × 2 ranks
  Wave 3 (``test_prepare_module_io_four_card_wave``):    10620–10621, two cases × 4 ranks

Coverage: **eight** scenarios on **2** NPUs + **two** scenarios on **4** NPUs,
including CPU / single-reference numerical checks (see worker docstrings).
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_prepare_module_io_distributed.py")


def _run_group(*cases):
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_prepare_module_io_two_card_wave_one():
    """
    Feature: first parallel_run — four 2-card worker cases (sum ranks = 8)
    Description:
        1. test_prepare_module_input_identity_roundtrip_npu
        2. test_prepare_module_output_replicate_to_shard_npu
        3. test_prepare_module_input_output_chain_npu
        4. test_prepare_module_input_then_colwise_linear_vs_cpu_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_prepare_module_input_identity_roundtrip_npu", 10600, 2),
        ("test_prepare_module_output_replicate_to_shard_npu", 10601, 2),
        ("test_prepare_module_input_output_chain_npu", 10602, 2),
        ("test_prepare_module_input_then_colwise_linear_vs_cpu_npu", 10603, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_prepare_module_io_two_card_wave_two():
    """
    Feature: second parallel_run — four 2-card worker cases (sum ranks = 8)
    Description:
        1. test_prepare_module_output_after_rowwise_vs_cpu_npu
        2. test_prepare_module_input_with_kwarg_scale_npu
        3. test_prepare_module_input_none_placeholder_dual_arg_npu
        4. test_prepare_module_output_tuple_with_none_slot_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_prepare_module_output_after_rowwise_vs_cpu_npu", 10610, 2),
        ("test_prepare_module_input_with_kwarg_scale_npu", 10611, 2),
        ("test_prepare_module_input_none_placeholder_dual_arg_npu", 10612, 2),
        ("test_prepare_module_output_tuple_with_none_slot_npu", 10613, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_prepare_module_io_four_card_wave():
    """
    Feature: third parallel_run — two 4-card worker cases (sum ranks = 8)
    Description:
        1. test_prepare_module_input_colwise_pipeline_vs_cpu_npu
        2. test_prepare_module_input_output_mlp_block_vs_cpu_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_prepare_module_input_colwise_pipeline_vs_cpu_npu", 10620, 4),
        ("test_prepare_module_input_output_mlp_block_vs_cpu_npu", 10621, 4),
    )
