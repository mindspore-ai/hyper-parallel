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
"""Pytest launchers for ``NoParallel`` NPU distributed tests.

Three launcher waves aligned with ``test_tp_styles_distributed.py`` convention
(one ``parallel_run`` per pytest test to avoid HCCL port conflicts):

* **Wave 1 — two-card wave one:** two cases × 2 ranks (ports 10560–10561).
* **Wave 2 — two-card wave two:** two cases × 2 ranks (ports 10562–10563).
* **Wave 3 — four-card wave:** two cases × 4 ranks (ports 10570–10571).

Worker implementations live in ``_test_no_parallel_distributed.py`` and compare
against a **CPU single-device** reference (float32).

Coverage:

1. Replicated Linear forward precision
2. Replicated LayerNorm forward precision
3. Input redistribution from Shard (SequenceParallel → NoParallel)
4. Output redistribution to Shard (NoParallel(output_layout=Shard(1)))
5. Replicated Linear backward gradient (4-GPU wave)
6. Composition: Colwise → NoParallel → Rowwise (4-GPU wave)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_no_parallel_distributed.py")


def _run_group(*cases):
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_no_parallel_two_card_wave_one():
    """
    Feature: two ``NoParallel`` scenarios on **2** NPU ranks (wave one)
    Description:
        1. test_no_parallel_linear_forward_precision_npu
        2. test_no_parallel_layernorm_forward_precision_npu
    Expectation: both worker cases succeed.
    """
    _run_group(
        ("test_no_parallel_linear_forward_precision_npu", 10560, 2),
        ("test_no_parallel_layernorm_forward_precision_npu", 10561, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_no_parallel_two_card_wave_two():
    """
    Feature: two ``NoParallel`` scenarios on **2** NPU ranks (wave two)
    Description:
        1. test_no_parallel_redistribute_sharded_input_npu
        2. test_no_parallel_redistribute_output_to_shard_npu
    Expectation: both worker cases succeed.
    """
    _run_group(
        ("test_no_parallel_redistribute_sharded_input_npu", 10562, 2),
        ("test_no_parallel_redistribute_output_to_shard_npu", 10563, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_no_parallel_four_card_wave():
    """
    Feature: two ``NoParallel`` scenarios on **4** NPU ranks
    Description:
        1. test_no_parallel_linear_backward_gradient_npu
        2. test_no_parallel_composition_sp_nopar_row_npu
    Expectation: both worker cases succeed.
    """
    _run_group(
        ("test_no_parallel_linear_backward_gradient_npu", 10570, 4),
        ("test_no_parallel_composition_sp_nopar_row_npu", 10571, 4),
    )
