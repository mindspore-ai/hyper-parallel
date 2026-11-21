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

The Level1 NPU wave runs Colwise and NoParallel concurrently on four ranks
with automatically assigned ports. The Gloo wave uses ports 10500–10502
and the second NPU wave uses 10510–10512.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_tp_styles_distributed.py")
_NO_PARALLEL_WORKER = str(Path(__file__).resolve().parent / "_test_no_parallel_distributed.py")


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
    Feature: Run two independent two-rank precision checks concurrently.
    Description:
        Compare Colwise Linear forward/backward and NoParallel Linear and redistribute.
        Exercise Colwise and Rowwise rejection paths in the Colwise NPU worker.
    Expectation: Outputs and gradients match the reference and both styles reject LayerNorm.
    """
    parallel_run([
        TorchCase(_WORKER, "test_colwise_precision_and_unsupported_styles_npu", num_proc=2),
        TorchCase(_NO_PARALLEL_WORKER, "test_no_parallel_linear_and_redistribute_npu", num_proc=2),
    ], global_num_proc=4)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_two_card_wave_one_gloo():
    """
    Feature: first parallel_run — three 2-card worker cases (gloo)
    Description:
        1. test_colwise_unsupported_module_raises_npu
        2. test_rowwise_unsupported_module_raises_npu
        3. test_colwise_linear_fwd_bwd_precision_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_colwise_unsupported_module_raises_npu", 10500, 2),
        ("test_rowwise_unsupported_module_raises_npu", 10501, 2),
        ("test_colwise_linear_fwd_bwd_precision_npu", 10502, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_styles_two_card_wave_two():
    """
    Feature: second parallel_run — three 2-card worker cases
    Description:
        1. test_rowwise_linear_fwd_bwd_precision_npu
        2. test_mlp_colwise_rowwise_forward_precision_npu
        3. test_colwise_embedding_forward_precision_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_rowwise_linear_fwd_bwd_precision_npu", 10510, 2),
        ("test_mlp_colwise_rowwise_forward_precision_npu", 10511, 2),
        ("test_colwise_embedding_forward_precision_npu", 10512, 2),
    )
