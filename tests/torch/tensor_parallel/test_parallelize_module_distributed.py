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

Functional coverage only (2-card). Col/Row Linear numerical precision lives in
``test_tp_styles_distributed`` — duplicate 4-card precision waves removed.

Port allocation:
  10460–10463  2-card functional
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_parallelize_module_distributed.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_functional_2card():
    """
    Feature: parallel_run launcher for 2-card parallelize_module functional coverage
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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_functional_2card_gloo():
    """
    Feature: parallel_run launcher for 2-card parallelize_module functional coverage
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
