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
"""Pytest launchers for ``parallelize_module`` NPU distributed tests.

Each launcher uses ``parallel_run`` to execute several worker cases from
``_test_parallelize_module_distributed.py`` concurrently on disjoint NPU subsets.

Typical packing:
  - up to four 2-card worker cases per launcher

Port allocation:
  10460–10469  2-card tests (single-node ``num_proc=2``)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

# Absolute path: torchrun inherits cwd (repo root or this dir); basename alone can fail.
_WORKER = str(Path(__file__).resolve().parent / "_test_parallelize_module_distributed.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_group1():
    """
    Feature: parallel_run launcher for 2-card parallelize_module coverage
    Description:
        1. test_parallelize_module_mesh_aligned_with_process_group_npu
        2. test_parallelize_module_dict_fnmatch_npu
        3. test_parallelize_module_src_data_rank_npu
        4. test_parallelize_module_single_style_root_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_parallelize_module_mesh_aligned_with_process_group_npu", 10460, 2),
        ("test_parallelize_module_dict_fnmatch_npu", 10461, 2),
        ("test_parallelize_module_src_data_rank_npu", 10462, 2),
        ("test_parallelize_module_single_style_root_npu", 10463, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_group2():
    """
    Feature: parallel_run launcher for 2-card parallelize_module precision coverage
    Description:
        1. test_parallelize_module_colwise_linear_precision_vs_pytorch_ref_npu
        2. test_parallelize_module_rowwise_linear_precision_vs_pytorch_ref_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_parallelize_module_colwise_linear_precision_vs_pytorch_ref_npu", 10464, 2),
        ("test_parallelize_module_rowwise_linear_precision_vs_pytorch_ref_npu", 10465, 2),
    )
