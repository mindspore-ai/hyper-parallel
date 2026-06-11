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
"""Pytest launchers for ``distribute_module`` NPU distributed integration tests.

Worker cases mirror PyTorch ``test/distributed/tensor/test_api.py``:
``DTensorAPITest.test_distribute_module`` and
``test_distribute_module_input_fn_output_fn``.

- **Functional**: four ``num_proc=2`` workers in one ``parallel_run`` (8 ranks).
- **Precision**: two ``num_proc=4`` workers in one ``parallel_run`` (colwise / rowwise
  ``distribute_module`` + ``F.linear`` CPU reference).

Port allocation:
  10470–10473  2-card functional
  10474–10475  4-card precision
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_distribute_module_distributed.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_distribute_module_group_npu():
    """
    Feature: parallel_run launcher for 2-card ``distribute_module`` integration tests
    Description:
        1. test_distribute_module_replicate_all_params_npu
        2. test_distribute_module_shard_all_linears_npu
        3. test_distribute_module_partial_shard_replicate_rest_npu
        4. test_distribute_module_input_output_hooks_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_distribute_module_replicate_all_params_npu", 10470, 2),
        ("test_distribute_module_shard_all_linears_npu", 10471, 2),
        ("test_distribute_module_partial_shard_replicate_rest_npu", 10472, 2),
        ("test_distribute_module_input_output_hooks_npu", 10473, 2),
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_distribute_module_group_npu_gloo():
    """
    Feature: parallel_run launcher for 2-card ``distribute_module`` integration tests
    Description:
        1. test_distribute_module_replicate_all_params_npu
        2. test_distribute_module_shard_all_linears_npu
        3. test_distribute_module_partial_shard_replicate_rest_npu
        4. test_distribute_module_input_output_hooks_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_distribute_module_replicate_all_params_npu", 10470, 2),
        ("test_distribute_module_shard_all_linears_npu", 10471, 2),
        ("test_distribute_module_partial_shard_replicate_rest_npu", 10472, 2),
        ("test_distribute_module_input_output_hooks_npu", 10473, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_distribute_module_precision_4card():
    """
    Feature: parallel_run launcher for 4-card ``distribute_module`` linear precision
    Description:
        1. test_distribute_module_colwise_linear_precision_vs_pytorch_ref_npu
        2. test_distribute_module_rowwise_linear_precision_vs_pytorch_ref_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_distribute_module_colwise_linear_precision_vs_pytorch_ref_npu", 10474, 4),
        ("test_distribute_module_rowwise_linear_precision_vs_pytorch_ref_npu", 10475, 4),
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_distribute_module_precision_4card_gloo():
    """
    Feature: parallel_run launcher for 4-card ``distribute_module`` linear precision
    Description:
        1. test_distribute_module_colwise_linear_precision_vs_pytorch_ref_npu
        2. test_distribute_module_rowwise_linear_precision_vs_pytorch_ref_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_distribute_module_colwise_linear_precision_vs_pytorch_ref_npu", 10474, 4),
        ("test_distribute_module_rowwise_linear_precision_vs_pytorch_ref_npu", 10475, 4),
    )
