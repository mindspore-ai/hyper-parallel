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
"""Distributed (PP=2) correctness tests for ScheduleMPipeTranspose.

Drives the per-rank worker ``_test_mpipe_transpose.py::test_mpipe_transpose``
through ``parallel_run`` / ``TorchCase`` (torchrun), once on NPU (hccl) and once
on CPU (gloo) — mirroring ``test_vpp_schedule.py``.  The worker runs both the
trainable-preprocess (T=2) and param-free dataload-only (T=0) paths and checks
the per-stage gradients + summed loss against a single-process reference.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

# Absolute path so torchrun+pytest finds the worker regardless of cwd.
_MPIPE_WORKER = str(Path(__file__).resolve().parent / "_test_mpipe_transpose.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_mpipe_transpose_dist():
    """
    Feature: parallel run case in pipeline_parallel.
    Description: Run ``test_mpipe_transpose`` (MPipe Transpose, PP=2, MB=4) across
        two NPU ranks — trainable-preprocess (T=2) then dataload-only (T=0).
    Expectation: per-stage parameter gradients and the summed loss match a
        single-process reference; run success.
    """
    parallel_run([
        TorchCase(_MPIPE_WORKER, "test_mpipe_transpose", num_proc=2)
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_mpipe_transpose_dist_gloo():
    """
    Feature: parallel run case in pipeline_parallel.
    Description: Run ``test_mpipe_transpose`` (MPipe Transpose, PP=2, MB=4) across
        two gloo/CPU ranks — trainable-preprocess (T=2) then dataload-only (T=0).
    Expectation: per-stage parameter gradients and the summed loss match a
        single-process reference; run success.
    """
    parallel_run([
        TorchCase(_MPIPE_WORKER, "test_mpipe_transpose", num_proc=2)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_mpipe_transpose_owner_backward_dist():
    """
    Feature: parallel run case in pipeline_parallel.
    Description: Run ``test_mpipe_transpose_owner_backward`` (MPipe Transpose,
        owner-does-backward, PP=2, MB=4, trainable T=2) across two NPU ranks.
    Expectation: every rank's reduced tower gradient + per-stage grads + summed
        loss match a single-process reference; run success.
    """
    parallel_run([
        TorchCase(_MPIPE_WORKER, "test_mpipe_transpose_owner_backward", num_proc=2)
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_mpipe_transpose_owner_backward_dist_gloo():
    """
    Feature: parallel run case in pipeline_parallel.
    Description: Run ``test_mpipe_transpose_owner_backward`` (MPipe Transpose,
        owner-does-backward, PP=2, MB=4, trainable T=2) across two gloo/CPU ranks.
    Expectation: every rank's reduced tower gradient + per-stage grads + summed
        loss match a single-process reference; run success.
    """
    parallel_run([
        TorchCase(_MPIPE_WORKER, "test_mpipe_transpose_owner_backward", num_proc=2)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_mpipe_transpose_owner_backward_accum_dist():
    """
    Feature: parallel run case in pipeline_parallel.
    Description: Run ``test_mpipe_transpose_owner_backward_accum`` (owner-backward,
        PP=2, 2 gradient-accumulation passes) across two NPU ranks.
    Expectation: the accumulated tower gradient matches the 2-pass reference (no
        re-reduce of earlier accumulation passes); run success.
    """
    parallel_run([
        TorchCase(_MPIPE_WORKER, "test_mpipe_transpose_owner_backward_accum", num_proc=2)
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_mpipe_transpose_owner_backward_accum_dist_gloo():
    """
    Feature: parallel run case in pipeline_parallel.
    Description: Run ``test_mpipe_transpose_owner_backward_accum`` (owner-backward,
        PP=2, 2 gradient-accumulation passes) across two gloo/CPU ranks.
    Expectation: the accumulated tower gradient matches the 2-pass reference (no
        re-reduce of earlier accumulation passes); run success.
    """
    parallel_run([
        TorchCase(_MPIPE_WORKER, "test_mpipe_transpose_owner_backward_accum", num_proc=2)
    ])
