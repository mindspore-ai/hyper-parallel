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

This file exposes **three** top-level pytest tests; each triggers one ``parallel_run``
(so three subprocess waves to torchrun/pytest). Worker cases live in
``_test_tp_styles_distributed.py`` and use ``dist.get_world_size()`` so the same
test body runs on 2- or 4-rank meshes as configured here.

Port allocation (unique per concurrent case within a wave; ``sum(num_proc) <= 8`` per wave):

  Wave 1 (``test_tp_styles_two_card_wave_one``): 10500–10503, four cases × 2 ranks
  Wave 2 (``test_tp_styles_two_card_wave_two``): 10510–10513, four cases × 2 ranks
  Wave 3 (``test_tp_styles_four_card_wave``):    10520–10521, two cases × 4 ranks

Coverage:

  * **Eight** worker scenarios run once on **2** NPUs (split 4+4 across the first two waves).
  * **Two** worker scenarios run again on **4** NPUs (linear col/row forward only, wider TP mesh).
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_two_card_wave_one():
    """
    Feature: first parallel_run — four 2-card worker cases (sum ranks = 8)
    Description:
        1. test_colwise_unsupported_module_raises_npu
        2. test_rowwise_unsupported_module_raises_npu
        3. test_colwise_linear_forward_precision_npu
        4. test_colwise_linear_backward_gradient_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_colwise_unsupported_module_raises_npu", 10500, 2),
        ("test_rowwise_unsupported_module_raises_npu", 10501, 2),
        ("test_colwise_linear_forward_precision_npu", 10502, 2),
        ("test_colwise_linear_backward_gradient_npu", 10503, 2),
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_two_card_wave_one_gloo():
    """
    Feature: first parallel_run — four 2-card worker cases (sum ranks = 8)
    Description:
        1. test_colwise_unsupported_module_raises_npu
        2. test_rowwise_unsupported_module_raises_npu
        3. test_colwise_linear_forward_precision_npu
        4. test_colwise_linear_backward_gradient_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_colwise_unsupported_module_raises_npu", 10500, 2),
        ("test_rowwise_unsupported_module_raises_npu", 10501, 2),
        ("test_colwise_linear_forward_precision_npu", 10502, 2),
        ("test_colwise_linear_backward_gradient_npu", 10503, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_two_card_wave_two():
    """
    Feature: second parallel_run — four 2-card worker cases (sum ranks = 8)
    Description:
        1. test_rowwise_linear_forward_precision_npu
        2. test_rowwise_linear_backward_gradient_npu
        3. test_mlp_colwise_rowwise_forward_precision_npu
        4. test_colwise_embedding_forward_precision_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_rowwise_linear_forward_precision_npu", 10510, 2),
        ("test_rowwise_linear_backward_gradient_npu", 10511, 2),
        ("test_mlp_colwise_rowwise_forward_precision_npu", 10512, 2),
        ("test_colwise_embedding_forward_precision_npu", 10513, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_four_card_wave():
    """
    Feature: third parallel_run — two 4-card worker cases (sum ranks = 8)
    Description:
        1. test_colwise_linear_forward_precision_npu (wider TP mesh)
        2. test_rowwise_linear_forward_precision_npu (wider TP mesh)
    Expectation: Run success.
    """
    _run_group(
        ("test_colwise_linear_forward_precision_npu", 10520, 4),
        ("test_rowwise_linear_forward_precision_npu", 10521, 4),
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_four_card_wave_gloo():
    """
    Feature: third parallel_run — two 4-card worker cases (sum ranks = 8)
    Description:
        1. test_colwise_linear_forward_precision_npu (wider TP mesh)
        2. test_rowwise_linear_forward_precision_npu (wider TP mesh)
    Expectation: Run success.
    """
    _run_group(
        ("test_colwise_linear_forward_precision_npu", 10520, 4),
        ("test_rowwise_linear_forward_precision_npu", 10521, 4),
    )
