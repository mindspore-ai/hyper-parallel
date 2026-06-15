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
"""Launcher for ``DTensor.copy_`` / ``zero_`` / ``fill_`` ST (gloo)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_dtensor_inplace.py")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_parallel_op_dtensor_inplace_group1_gloo():
    """
    Feature: DTensor in-place ops (copy_/zero_/fill_).
    Description:
        1. test_dtensor_copy_same_placement
        2. test_dtensor_copy_dtype_cast
        3. test_dtensor_copy_scalar_broadcast
        4. test_dtensor_zero
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_dtensor_copy_same_placement", num_proc=2),
        TorchCase(IMPL_FILE, "test_dtensor_copy_dtype_cast", num_proc=2),
        TorchCase(IMPL_FILE, "test_dtensor_copy_scalar_broadcast", num_proc=2),
        TorchCase(IMPL_FILE, "test_dtensor_zero", num_proc=2),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_parallel_op_dtensor_inplace_group2_gloo():
    """
    Feature: DTensor in-place ops (copy_/zero_/fill_).
    Description:
        1. test_dtensor_fill
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_dtensor_fill", num_proc=2),
    ])
