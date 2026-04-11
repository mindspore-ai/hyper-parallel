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
"""Pytest launchers for ``ColwiseParallel`` / ``RowwiseParallel`` NPU distributed tests.

Uses the real ``ColwiseParallel`` / ``RowwiseParallel`` styles from
``hyper_parallel.core.tensor_parallel.style`` (not custom substitutes).

Port allocation:
  10500–10501  2-card functional (unsupported module rejection)
  10502–10505  4-card precision (colwise/rowwise linear forward + backward)
  10506        4-card precision (MLP colwise+rowwise composition)
  10507        4-card precision (colwise embedding)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_tp_styles_distributed.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_functional_2card():
    """
    Feature: parallel_run launcher for 2-card functional coverage
    Description:
        1. test_colwise_unsupported_module_raises_npu
        2. test_rowwise_unsupported_module_raises_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_colwise_unsupported_module_raises_npu", 10500, 2),
        ("test_rowwise_unsupported_module_raises_npu", 10501, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_linear_precision_4card():
    """
    Feature: parallel_run launcher for 4-card Linear precision vs CPU reference
    Description:
        1. test_colwise_linear_forward_precision_npu
        2. test_colwise_linear_backward_gradient_npu
        3. test_rowwise_linear_forward_precision_npu
        4. test_rowwise_linear_backward_gradient_npu
    Expectation: Run success.
    """
    # Split into two groups to keep total processes ≤ 8 cards per group
    _run_group(
        ("test_colwise_linear_forward_precision_npu", 10502, 4),
        ("test_colwise_linear_backward_gradient_npu", 10503, 4),
    )
    _run_group(
        ("test_rowwise_linear_forward_precision_npu", 10504, 4),
        ("test_rowwise_linear_backward_gradient_npu", 10505, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_mlp_composition_4card():
    """
    Feature: parallel_run launcher for 4-card MLP composition (colwise + rowwise)
    Description:
        1. test_mlp_colwise_rowwise_forward_precision_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_mlp_colwise_rowwise_forward_precision_npu", 10506, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_embedding_precision_4card():
    """
    Feature: parallel_run launcher for 4-card Embedding precision vs CPU reference
    Description:
        1. test_colwise_embedding_forward_precision_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_colwise_embedding_forward_precision_npu", 10507, 4),
    )
