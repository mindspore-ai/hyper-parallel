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
"""Pytest launchers for ``SequenceParallel`` NPU distributed tests.

Two launcher waves (two ``parallel_run`` workers):

* **Wave 1 — dual GPU:** four cases × 2 ranks (ports 10540–10543), sum ranks = 8.
* **Wave 2 — quad GPU:** two cases × 4 ranks (ports 10550–10551), sum ranks = 8.

Worker implementations compare against a **CPU single-device** reference (float32).
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_tp_sequence_parallel_distributed.py")


def _run_dual_gpu_wave():
    parallel_run([
        TorchCase(_WORKER, "test_sequence_parallel_layernorm_forward_chunk_vs_cpu_2gpu", 10540, 2),
        TorchCase(_WORKER, "test_sequence_parallel_layernorm_forward_gather_full_vs_cpu_2gpu", 10541, 2),
        TorchCase(_WORKER, "test_sequence_parallel_dropout_identity_vs_cpu_2gpu", 10542, 2),
        TorchCase(_WORKER, "test_sequence_parallel_layernorm_no_affine_forward_vs_cpu_2gpu", 10543, 2),
    ])


def _run_quad_gpu_wave():
    parallel_run([
        TorchCase(_WORKER, "test_sequence_parallel_layernorm_forward_gather_full_vs_cpu_4gpu", 10550, 4),
        TorchCase(_WORKER, "test_sequence_parallel_layernorm_backward_grad_vs_cpu_4gpu", 10551, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_sequence_parallel_dual_gpu_wave():
    """
    Feature: four ``SequenceParallel`` scenarios on **2** NPU ranks
    Expectation: all four worker cases succeed.
    """
    _run_dual_gpu_wave()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_sequence_parallel_quad_gpu_wave():
    """
    Feature: two ``SequenceParallel`` scenarios on **4** NPU ranks
    Expectation: both worker cases succeed.
    """
    _run_quad_gpu_wave()
